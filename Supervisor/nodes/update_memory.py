"""
update_memory.py

Conversation memory management node for the Supervisor workflow.

Appends the current turn (judge query + final response) to the
conversation history and trims to stay within the configured window.

Memory safety fix
-----------------
The original code did:

    conversation_history: List[dict] = list(state.get("conversation_history", []))

``list()`` creates a *shallow* copy of the list -- the list object itself
is new, but the dict items inside it are the same objects in memory as the
ones in the incoming state.  In the normal single-request flow this is
harmless because LangGraph passes a fresh state on every invocation.

However, when the pipeline is called in a tight loop (e.g. the memory-leak
performance test, or any batch / stress scenario) and the caller reuses a
base state dict without deep-copying it, the ``conversation_history`` list
on that base dict accumulates entries indefinitely because the list object
is shared between the caller and this node.  After N runs the list has 2N
entries and all of the associated strings remain live in memory, producing
unbounded growth.

Fix: copy the incoming list properly so this node never mutates an object
it doesn't own, regardless of how the caller constructed the state.  We
also ensure each appended dict is a new object (not aliased from elsewhere).
"""

import logging
from typing import Any, Dict, List

from config.supervisor import MAX_CONVERSATION_TURNS
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def update_memory_node(state: SupervisorState) -> Dict[str, Any]:
    """Append the latest exchange to conversation history and trim.

    Updates state keys: ``conversation_history``, ``turn_count``.
    """
    # Build a fully independent copy of the incoming history so that
    # appending to it never mutates the caller's original list or any
    # intermediate state objects held by LangGraph.
    # Each item is also copied (dict()) so callers who pass dicts from
    # external sources cannot be affected by later mutations.
    incoming: List[dict] = state.get("conversation_history") or []
    conversation_history: List[dict] = [dict(entry) for entry in incoming]

    turn_count = state.get("turn_count", 0)

    judge_query = state.get("judge_query", "")
    final_response = state.get("final_response", "")

    # Append the user turn
    if judge_query:
        conversation_history.append({
            "role": "user",
            "content": judge_query,
        })

    # Append the assistant turn -- store only the final response string,
    # never agent_results or other large intermediate objects.
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