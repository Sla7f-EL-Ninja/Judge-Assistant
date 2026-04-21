"""
executor.py

Executor nodes for the Chat Reasoner fan-out/fan-in execution wave.

executor_fanout_node  — pass-through; routing handled by executor_dispatch_router.
executor_dispatch_router — conditional edge function; returns list of Send objects
                           for parallel step execution, or "collector" if no steps ready.
step_worker_node      — runs one tool step (called via LangGraph Send).
collector_node        — fan-in pass-through; routing handled by collector_router.
collector_router      — decides: more waves | synthesizer | replanner | trace_writer.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from langgraph.types import Send

from chat_reasoner.state import (
    ChatReasonerState,
    StepResult,
    StepWorkerPayload,
)
from chat_reasoner.tools import dispatch_tool

logger = logging.getLogger(__name__)

_STEP_FAILURE_CAP = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_ready_steps(state: ChatReasonerState) -> List[dict]:
    """Return plan steps that are ready to execute this wave.

    A step is ready when:
    - Its depends_on steps all have status success or skipped, AND
    - It has no existing success/skipped result, AND
    - It has not hit the failure cap.
    """
    completed_ids = {
        r["step_id"]
        for r in state.get("step_results", [])
        if r.get("status") in ("success", "skipped")
    }
    failed_ids = {
        r["step_id"]
        for r in state.get("step_results", [])
        if r.get("status") == "failure"
    }
    failure_counts = state.get("step_failures", {})
    capped_ids = {sid for sid, cnt in failure_counts.items() if cnt >= _STEP_FAILURE_CAP}

    # Steps in flight (have a failure result but under cap) should be retried
    plan = state.get("plan", [])
    ready = []
    for step in plan:
        sid = step["step_id"]
        if sid in completed_ids:
            continue
        if sid in capped_ids:
            continue
        deps = step.get("depends_on", [])
        if all(d in completed_ids for d in deps):
            ready.append(step)
    return ready


def _all_steps_terminal(state: ChatReasonerState) -> bool:
    """True when every plan step has a success, skipped, or capped-failure result."""
    capped = {
        sid for sid, cnt in state.get("step_failures", {}).items()
        if cnt >= _STEP_FAILURE_CAP
    }
    completed = {
        r["step_id"]
        for r in state.get("step_results", [])
        if r.get("status") in ("success", "skipped")
    }
    for step in state.get("plan", []):
        sid = step["step_id"]
        if sid not in completed and sid not in capped:
            return False
    return True


def _first_capped_failure(state: ChatReasonerState) -> tuple:
    """Return (step_id, last_error) for the first step that hit the failure cap,
    or (None, None) if none."""
    capped = {
        sid: cnt for sid, cnt in state.get("step_failures", {}).items()
        if cnt >= _STEP_FAILURE_CAP
    }
    if not capped:
        return None, None
    sid = next(iter(capped))
    # Find last error for this step from results
    last_error = ""
    for r in reversed(state.get("step_results", [])):
        if r.get("step_id") == sid and r.get("status") == "failure":
            last_error = r.get("error", "")
            break
    return sid, last_error


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def executor_fanout_node(state: ChatReasonerState) -> Dict[str, Any]:
    """Pass-through node. Routing to step_workers happens in the conditional edge."""
    return {}


def executor_dispatch_router(state: ChatReasonerState):
    """Conditional edge: fan out with Send or go to collector if nothing ready."""
    ready = _find_ready_steps(state)
    if not ready:
        logger.debug("executor_dispatch: no ready steps → collector")
        return "collector"

    logger.info("executor_dispatch: fanning out %d steps", len(ready))
    completed_results = {
        r["step_id"]: r
        for r in state.get("step_results", [])
        if r.get("status") in ("success", "skipped")
    }
    return [
        Send(
            "step_worker",
            {
                "step": step,
                "case_id": state.get("case_id", ""),
                "conversation_history": state.get("conversation_history", []),
                "prior_results": [
                    completed_results[dep]
                    for dep in step.get("depends_on", [])
                    if dep in completed_results
                ],
            },
        )
        for step in ready
    ]


def step_worker_node(payload: StepWorkerPayload) -> Dict[str, Any]:
    """Execute one tool step. Receives a Send payload, not the full graph state.

    Returns state updates that are merged into ChatReasonerState via reducers:
    - step_results (Annotated[list, add])
    - step_failures (Annotated[dict, _merge_step_failures])
    - tool_calls_log (Annotated[list, add])
    """
    step = payload["step"]
    case_id = payload["case_id"]
    conversation_history = payload["conversation_history"]
    prior_results = payload.get("prior_results", [])
    step_id = step.get("step_id", "?")

    logger.info("step_worker: executing step=%s tool=%s", step_id, step.get("tool"))

    result: StepResult = dispatch_tool(step, case_id, conversation_history, prior_results)

    log_entry = {
        "step_id": step_id,
        "tool": step.get("tool"),
        "query": step.get("query"),
        "status": result.status,
        "response_preview": result.response[:200] if result.response else "",
        "sources": result.sources,
        "error": result.error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    new_failures: Dict[str, int] = {}
    if result.status == "failure":
        logger.warning("step_worker: step %s failed — %s", step_id, result.error)
        new_failures[step_id] = 1  # reducer will accumulate via _merge_step_failures

    logger.info("step_worker: step=%s status=%s", step_id, result.status)
    return {
        "step_results": [result.model_dump()],
        "step_failures": new_failures,
        "tool_calls_log": [log_entry],
    }


def collector_node(state: ChatReasonerState) -> Dict[str, Any]:
    """Fan-in: set replan trigger fields if a step hit its failure cap."""
    capped_sid, capped_error = _first_capped_failure(state)
    if capped_sid is not None and state.get("replan_count", 0) < 2:
        return {
            "replan_trigger_step_id": capped_sid,
            "replan_trigger_error": capped_error or "consecutive failures",
        }
    if capped_sid is not None:
        return {
            "status": "failed",
            "error_message": "تعذّر إتمام خطة الاستدلال بعد استنفاد محاولات إعادة التخطيط.",
        }
    return {}


def collector_router(state: ChatReasonerState) -> str:
    """Route after collector_node has updated trigger fields."""
    if state.get("status") == "failed":
        logger.error("collector: replan cap reached → trace_writer (failed)")
        return "trace_writer"

    capped_sid, _ = _first_capped_failure(state)
    if capped_sid is not None:
        logger.warning("collector: step %s hit failure cap → replanner", capped_sid)
        return "replanner"

    if not _all_steps_terminal(state):
        logger.debug("collector: pending steps remain → executor_fanout")
        return "executor_fanout"

    logger.info("collector: all steps terminal → synthesizer")
    return "synthesizer"
