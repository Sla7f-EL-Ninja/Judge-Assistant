"""
fallback.py

Fallback response node for the Supervisor workflow.

Triggered when validation fails after exhausting all retry attempts.
Returns a response explaining the limitation and includes the validation
feedback so the judge can adjust the query.
"""

from typing import Any, Dict

from Supervisor.prompts import FALLBACK_RESPONSE_TEMPLATE
from Supervisor.state import SupervisorState


def fallback_response_node(state: SupervisorState) -> Dict[str, Any]:
    """Return a fallback response with validation feedback.

    Sets ``final_response`` so the memory node can record it.
    """
    feedback = state.get("validation_feedback", "")
    fallback_text = FALLBACK_RESPONSE_TEMPLATE.format(feedback=feedback)

    return {
        "final_response": fallback_text,
        "validation_status": "fallback",
    }
