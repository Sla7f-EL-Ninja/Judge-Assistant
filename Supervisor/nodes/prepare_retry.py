"""
prepare_retry.py

Pre-retry node — enforces exponential backoff between validation failures
and the next agent dispatch cycle (A6.3.2, A6.6.4, A6.7).

Isolates retry-specific logic (logging + sleep) from dispatch_agents_node,
which now only handles first-dispatch and re-dispatch — not backoff timing.
"""

import logging
import time
from typing import Any, Dict

from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

_BASE_BACKOFF_S: float = 2.0
_MAX_BACKOFF_S: float = 10.0


def prepare_retry_node(state: SupervisorState) -> Dict[str, Any]:
    """Sleep a brief exponential backoff before re-dispatching agents.

    validate_output already incremented retry_count before routing here,
    so retry_count >= 1 when this node runs.  Returns an empty dict —
    no state mutation; this node only enforces the cooldown.
    """
    retry_count = state.get("retry_count", 1)
    max_retries = state.get("max_retries", 3)
    validation_status = state.get("validation_status", "unknown")

    backoff_s = min(_BASE_BACKOFF_S * (2 ** (retry_count - 1)), _MAX_BACKOFF_S)

    logger.info(
        "Retry %d/%d — sleeping %.1fs before re-dispatch (status=%s cid=%s)",
        retry_count,
        max_retries,
        backoff_s,
        validation_status,
        state.get("correlation_id", ""),
    )

    time.sleep(backoff_s)
    return {}
