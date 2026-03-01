"""
update_memory.py

Conversation memory management node for the Supervisor workflow.

Appends the current turn (judge query + final response) to the
conversation history and trims to stay within the configured window.
"""

import logging
from typing import Any, Dict, List

from Supervisor.config import MAX_CONVERSATION_TURNS
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def update_memory_node(state: SupervisorState) -> Dict[str, Any]:
    """Append the latest exchange to conversation history and trim.

    Updates state keys: ``conversation_history``, ``turn_count``.
    """
    conversation_history: List[dict] = list(
        state.get("conversation_history", [])
    )
    turn_count = state.get("turn_count", 0)

    judge_query = state.get("judge_query", "")
    final_response = state.get("final_response", "")

    # Append the user turn
    if judge_query:
        conversation_history.append({
            "role": "user",
            "content": judge_query,
        })

    # Append the assistant turn
    if final_response:
        conversation_history.append({
            "role": "assistant",
            "content": final_response,
        })

    turn_count += 1

    # Trim to the configured maximum (each turn = 2 messages)
    max_messages = MAX_CONVERSATION_TURNS * 2
    if len(conversation_history) > max_messages:
        conversation_history = conversation_history[-max_messages:]

    logger.info(
        "Memory updated: turn=%d, history_len=%d",
        turn_count,
        len(conversation_history),
    )

    return {
        "conversation_history": conversation_history,
        "turn_count": turn_count,
    }
