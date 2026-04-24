"""
validate_output.py

Output validation node for the Supervisor workflow.

Runs four quality checks (hallucination, relevance, completeness, coherence)
on the merged response before it reaches the judge.

G5.7.6 partial-pass path: when hallucination, relevance, and coherence pass
but completeness alone fails, the answer is accepted with a disclosure caveat
instead of triggering a retry.  This avoids the all-or-nothing retry cost for
responses that are factually sound but merely incomplete.

G5.7.4 cross-turn coherence: the prior assistant turn (if any) is included in
the validator prompt so the LLM can detect direct contradictions between turns.
"""

import json
import logging
from typing import Any, Dict

from config import get_llm
from Supervisor.llm_utils import llm_invoke
from Supervisor.metrics import RETRY_COUNTER, TURN_COUNTER
from Supervisor.prompts import (
    PRIOR_RESPONSE_SECTION_EMPTY,
    PRIOR_RESPONSE_SECTION_TEMPLATE,
    VALIDATION_SYSTEM_PROMPT,
    VALIDATION_USER_TEMPLATE,
)
from Supervisor.state import SupervisorState, ValidationResult

logger = logging.getLogger(__name__)

# Disclosure caveat appended to partial-pass responses (G5.7.6)
_PARTIAL_PASS_CAVEAT = (
    "\n\n---\n"
    "**ملاحظة:** قد لا تكون هذه الإجابة شاملة لجميع جوانب السؤال. "
    "يُنصح بإعادة الاستفسار عن الجوانب التي لم تُعالَج بشكل كافٍ."
)

# G5.8.1: PII (names, national IDs, addresses) in judge_query and case docs is
# forwarded to the external Gemini API.  A PII-redaction layer must be added
# before these LLM calls for production judicial deployments.  Track as a
# separate compliance task; not implemented here to avoid scope creep.


def _extract_prior_response(state: SupervisorState) -> str:
    """Return the last assistant response from conversation_history, or ''."""
    history = state.get("conversation_history") or []
    for entry in reversed(history):
        if entry.get("role") == "assistant":
            content = entry.get("content", "")
            if content:
                # Truncate very long prior responses to keep prompt size bounded
                return content[:2000] + ("..." if len(content) > 2000 else "")
    return ""


def validate_output_node(state: SupervisorState) -> Dict[str, Any]:
    """Validate the merged response against the four quality criteria.

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

    # Build a summary of raw agent outputs for the validator.
    # Include raw_output (retrieved content) so hallucination check has
    # access to the actual source material, not just the formatted response.
    raw_parts = []
    for agent_name, result in agent_results.items():
        response = result.get("response", "")
        raw_output = result.get("raw_output", {})
        section = [f"--- {agent_name} ---", response]
        if raw_output:
            try:
                raw_str = json.dumps(raw_output, ensure_ascii=False, default=str)
                if len(raw_str) > 3000:
                    raw_str = raw_str[:3000] + "...[truncated]"
                section.append(f"[raw: {raw_str}]")
            except Exception:
                pass
        raw_parts.append("\n".join(section))
    raw_outputs_text = "\n\n".join(raw_parts) if raw_parts else "(no raw outputs)"

    # Build prior-response section for coherence check (G5.7.4)
    prior_response = _extract_prior_response(state)
    if prior_response:
        prior_response_section = PRIOR_RESPONSE_SECTION_TEMPLATE.format(
            prior_response=prior_response
        )
    else:
        prior_response_section = PRIOR_RESPONSE_SECTION_EMPTY

    user_prompt = VALIDATION_USER_TEMPLATE.format(
        judge_query=judge_query,
        raw_agent_outputs=raw_outputs_text,
        prior_response_section=prior_response_section,
        response=merged_response,
    )

    try:
        llm = get_llm("low")
        structured_llm = llm.with_structured_output(ValidationResult)

        messages = [
            {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result = llm_invoke(structured_llm.invoke, messages)

        if not result:
            raise ValueError("Validation model returned None.")

        if not hasattr(result, "overall_pass"):
            raise ValueError("Validation result missing required fields.")

        # G5.7.6 — Partial-pass: hallucination + relevance + coherence pass but
        # completeness fails.  Accept the answer with a disclosure caveat instead
        # of burning a retry on an incomplete-but-sound response.
        hallucination_ok = getattr(result, "hallucination_pass", False)
        relevance_ok = getattr(result, "relevance_pass", False)
        completeness_ok = getattr(result, "completeness_pass", False)
        coherence_ok = getattr(result, "coherence_pass", True)

        if hallucination_ok and relevance_ok and coherence_ok and not completeness_ok:
            logger.info(
                "Partial-pass: hallucination+relevance+coherence OK, completeness failed — "
                "accepting with disclosure caveat (G5.7.6)"
            )
            TURN_COUNTER.labels(status="partial_pass").inc()
            return {
                "validation_status": "partial_pass",
                "validation_feedback": result.feedback,
                "final_response": merged_response + _PARTIAL_PASS_CAVEAT,
            }

        if result.overall_pass:
            TURN_COUNTER.labels(status="pass").inc()
            return {
                "validation_status": "pass",
                "validation_feedback": "",
                "final_response": merged_response,
            }

        # Full failure — determine dominant failure type for routing
        if not hallucination_ok:
            status = "fail_hallucination"
        elif not coherence_ok:
            status = "fail_hallucination"  # coherence failure is a correctness issue
        elif not relevance_ok:
            status = "fail_relevance"
        else:
            status = "fail_completeness"

        RETRY_COUNTER.labels(reason=status).inc()
        return {
            "validation_status": status,
            "validation_feedback": result.feedback,
            "retry_count": retry_count + 1,
        }

    except Exception as exc:
        logger.exception("Validation LLM error (distinct from content failure): %s", exc)
        return {
            "validation_status": "validator_error",
            "validation_feedback": f"Validator unavailable: {exc}",
            "retry_count": retry_count + 1,
        }
