"""
merge_responses.py

Multi-agent response merger node for the Supervisor workflow.

When multiple agents were invoked, this node uses an LLM to synthesise
their outputs into a single coherent response.  For single-agent results
it simply passes the response through.
"""

import logging
from typing import Any, Dict, List

from config import get_llm
from Supervisor.llm_utils import llm_invoke
from Supervisor.prompts import (
    MERGE_RESPONSES_SYSTEM_PROMPT,
    MERGE_RESPONSES_USER_TEMPLATE,
)
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def merge_responses_node(state: SupervisorState) -> Dict[str, Any]:
    """Merge agent outputs into a single ``merged_response``.

    Also aggregates ``sources`` from all agents.
    """
    agent_results = state.get("agent_results", {})
    agent_errors = state.get("agent_errors", {})

    if not agent_results:
        # All agents failed -- nothing to merge; validate_output will set validation_status
        error_summary = "; ".join(
            f"{k}: {v}" for k, v in agent_errors.items()
        )
        logger.warning("All agents failed: %s", error_summary)
        return {
            "merged_response": "",
            "sources": [],
            "validation_feedback": f"All agents failed: {error_summary}",
        }

    # Collect and normalize sources from every agent (P1.6.8)
    all_sources: List[str] = []
    for result in agent_results.values():
        for src in result.get("sources", []):
            if src:
                all_sources.append(str(src).strip())
    # Deduplicate while preserving order (case-insensitive key)
    seen: set = set()
    unique_sources = []
    for src in all_sources:
        key = src.lower()
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    # Single agent -- pass through directly
    if len(agent_results) == 1:
        single_result = next(iter(agent_results.values()))
        return {
            "merged_response": single_result.get("response", ""),
            "sources": unique_sources,
        }

    # Multiple agents -- use LLM to merge
    judge_query = state.get("classified_query", state.get("judge_query", ""))

    agent_output_parts = []
    for agent_name, result in agent_results.items():
        response = result.get("response", "")
        agent_output_parts.append(f"--- {agent_name} ---\n{response}")
    agent_outputs_text = "\n\n".join(agent_output_parts)

    user_prompt = MERGE_RESPONSES_USER_TEMPLATE.format(
        judge_query=judge_query,
        agent_outputs=agent_outputs_text,
    )

    try:
        llm = get_llm("high")
        messages = [
            {"role": "system", "content": MERGE_RESPONSES_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = llm_invoke(llm.invoke, messages)
        if response is None:
            raise ValueError("LLM returned None for merge_responses")
        merged = response.content if hasattr(response, "content") else str(response)

        return {
            "merged_response": merged,
            "sources": unique_sources,
        }

    except Exception as exc:
        logger.exception("Response merge failed: %s", exc)
        # Fallback: concatenate responses
        fallback = "\n\n".join(
            f"[{name}]\n{r.get('response', '')}"
            for name, r in agent_results.items()
        )
        return {
            "merged_response": fallback,
            "sources": unique_sources,
        }
