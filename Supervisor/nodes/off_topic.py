"""
off_topic.py

Off-topic response node for the Supervisor workflow.

Returns a polite message when the judge query is unrelated to civil law.
"""

import logging
from typing import Any, Dict

from Supervisor.metrics import TURN_COUNTER
from Supervisor.prompts import OFF_TOPIC_RESPONSE
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def off_topic_response_node(state: SupervisorState) -> Dict[str, Any]:
    """Return a canned off-topic response.

    Sets ``final_response`` so the memory node can record it.
    """
    logger.info("Off-topic response returned. intent=%r query=%r", state.get("intent"), state.get("judge_query", "")[:100])
    TURN_COUNTER.labels(status="off_topic").inc()
    return {
        "final_response": OFF_TOPIC_RESPONSE,
        "merged_response": OFF_TOPIC_RESPONSE,
        "validation_status": "off_topic",
    }
