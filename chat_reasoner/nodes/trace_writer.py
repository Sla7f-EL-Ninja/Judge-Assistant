"""
trace_writer.py

Terminal node: always runs last. Writes trace to MongoDB (best-effort) and
sets status to "succeeded" if not already "failed".
"""

import logging
from typing import Any, Dict

from chat_reasoner.state import ChatReasonerState
from chat_reasoner.trace import write_trace

logger = logging.getLogger(__name__)


def trace_writer_node(state: ChatReasonerState) -> Dict[str, Any]:
    current_status = state.get("status", "running")

    # Determine final status
    final_status = "failed" if current_status == "failed" else "succeeded"

    # Write trace (best-effort; write_trace swallows all exceptions internally)
    write_trace({**state, "status": final_status})

    logger.info("trace_writer: run complete — status=%s", final_status)
    return {"status": final_status}
