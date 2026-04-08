"""
off_topic.py

Off-topic response node for the Supervisor workflow.

Returns a polite message when the judge query is unrelated to civil law.
"""

from typing import Any, Dict

from Supervisor.prompts import OFF_TOPIC_RESPONSE
from Supervisor.state import SupervisorState


def off_topic_response_node(state: SupervisorState) -> Dict[str, Any]:
    """Return a canned off-topic response.

    Sets ``final_response`` so the memory node can record it.
    """
    return {
        "final_response": OFF_TOPIC_RESPONSE,
        "merged_response": OFF_TOPIC_RESPONSE,
        "validation_status": "pass",
    }
