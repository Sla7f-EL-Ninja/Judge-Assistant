"""
write_long_term_memory.py

Graph node that persists long-term memories after a successful turn.

Runs on the pass branch only (validate_output → update_memory → this node).
Off-topic and fallback branches bypass this to avoid noisy memory writes
from rejected or failed turns.

Semantic memory: extracted synchronously (must be persisted immediately so
the next turn can retrieve it).

Episodic + procedural memory: scheduled via ReflectionExecutor (background,
fire-and-forget) so this node returns quickly.

All store / reflection calls are wrapped — failures log a warning and the
graph continues.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from config.supervisor import EPISODIC_REFLECT_DELAY_S
from Supervisor.services.memory import (
    get_episodic_manager,
    get_procedural_manager,
    get_reflection_executor,
    get_semantic_manager,
    get_store,
)
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def write_long_term_memory_node(state: SupervisorState) -> Dict[str, Any]:
    """Persist semantic facts synchronously; schedule episodic/procedural async."""
    case_id: str = state.get("case_id", "")
    user_id: Optional[str] = state.get("user_id")
    judge_query: str = state.get("judge_query", "")
    final_response: str = state.get("final_response", "")
    semantic_facts: List[Dict] = state.get("semantic_facts") or []

    validation_status = state.get("validation_status") or ""
    if validation_status not in ("pass", "partial_pass"):
        return {}

    if not judge_query or not final_response:
        return {}

    messages = [
        {"role": "user", "content": judge_query},
        {"role": "assistant", "content": final_response},
    ]

    # -- Semantic: synchronous write --
    if case_id:
        try:
            manager = get_semantic_manager(case_id)
            manager.invoke({"messages": messages, "existing": semantic_facts})
            logger.info("write_long_term_memory: semantic facts updated for case %s", case_id)
        except Exception as exc:
            logger.warning("write_long_term_memory: semantic write failed — %s", exc)

    # -- Episodic: background reflection --
    if case_id:
        try:
            executor = get_reflection_executor()
            executor.schedule(
                get_episodic_manager(case_id),
                {"messages": messages},
                after=EPISODIC_REFLECT_DELAY_S,
            )
            logger.debug("write_long_term_memory: episodic reflection scheduled for case %s", case_id)
        except Exception as exc:
            logger.warning("write_long_term_memory: episodic schedule failed — %s", exc)

    # -- Procedural: background reflection --
    if user_id:
        try:
            executor = get_reflection_executor()
            executor.schedule(
                get_procedural_manager(user_id),
                {"messages": messages},
                after=EPISODIC_REFLECT_DELAY_S,
            )
            logger.debug("write_long_term_memory: procedural reflection scheduled for user %s", user_id)
        except Exception as exc:
            logger.warning("write_long_term_memory: procedural schedule failed — %s", exc)

    return {}
