"""
validate_output.py

Output validation node for the Supervisor workflow.

Runs three quality checks (hallucination, relevance, completeness) on
the merged response before it reaches the judge.
"""

import logging
from typing import Any, Dict

from config import get_llm
from Supervisor.prompts import VALIDATION_SYSTEM_PROMPT, VALIDATION_USER_TEMPLATE
from Supervisor.state import SupervisorState, ValidationResult

logger = logging.getLogger(__name__)


def validate_output_node(state: SupervisorState) -> Dict[str, Any]:
    """Validate the merged response against the three quality criteria.

    Updates state keys: ``validation_status``, ``validation_feedback``,
    ``retry_count``, ``final_response``.
    """
    merged_response = state.get("merged_response", "")
    judge_query = state.get("classified_query", state.get("judge_query", ""))
    agent_results = state.get("agent_results", {})
    retry_count = state.get("retry_count", 0)

    if not merged_response:
        return {
            "validation_status": "fail_completeness",
            "validation_feedback": "No response was generated to validate.",
            "retry_count": retry_count + 1,
        }

    # Build a summary of raw agent outputs for the validator
    raw_parts = []
    for agent_name, result in agent_results.items():
        raw_output = result.get("raw_output", {})
        response = result.get("response", "")
        raw_parts.append(f"--- {agent_name} ---\n{response}")
    raw_outputs_text = "\n\n".join(raw_parts) if raw_parts else "(no raw outputs)"

    user_prompt = VALIDATION_USER_TEMPLATE.format(
        judge_query=judge_query,
        raw_agent_outputs=raw_outputs_text,
        response=merged_response,
    )

    try:
        llm = get_llm("low")
        structured_llm = llm.with_structured_output(ValidationResult)

        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result = structured_llm.invoke(messages)

        if not result:
            raise ValueError("Validation model returned None.")

        if not hasattr(result, "overall_pass"):
            raise ValueError("Validation result missing required fields.")

        if result.overall_pass:
            return {
                "validation_status": "pass",
                "validation_feedback": "",
                "final_response": merged_response,
            }

        if not result.hallucination_pass:
            status = "fail_hallucination"
        elif not result.relevance_pass:
            status = "fail_relevance"
        else:
            status = "fail_completeness"

        return {
            "validation_status": status,
            "validation_feedback": result.feedback,
            "retry_count": retry_count + 1,
        }

    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        # If validation itself fails, let the response through with a warning
        return {
            "validation_status": "pass",
            "validation_feedback": f"Validation skipped due to error: {exc}",
            "final_response": merged_response,
        }
