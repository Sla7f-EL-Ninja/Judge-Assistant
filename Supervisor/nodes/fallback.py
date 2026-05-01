"""
fallback.py

Fallback response node for the Supervisor workflow.

Triggered when validation fails after exhausting all retry attempts.
Returns a response explaining the limitation and includes the validation
feedback so the judge can adjust the query.
"""

import logging
from typing import Any, Dict

from Supervisor.metrics import FALLBACK_COUNTER, TURN_COUNTER
from Supervisor.prompts import FALLBACK_RESPONSE_TEMPLATE
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def fallback_response_node(state: SupervisorState) -> Dict[str, Any]:
    """Return a fallback response with validation feedback.

    Sets ``final_response`` so the memory node can record it.
    """
    feedback = state.get("validation_feedback", "")
    logger.info("Fallback triggered after %d retries. feedback=%r", state.get("retry_count", 0), feedback[:200])
    FALLBACK_COUNTER.inc()
    TURN_COUNTER.labels(status="fallback").inc()
    fallback_text = FALLBACK_RESPONSE_TEMPLATE.format(feedback=feedback)

    return {
        "final_response": fallback_text,
        "validation_status": "fallback",
    }
