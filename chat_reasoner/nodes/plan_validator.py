"""
plan_validator.py

Pure-Python plan validator — no LLM involved.
Runs 6 structural checks and returns PlanValidationResult.
Routing: valid → executor_fanout; invalid (retry < 3) → planner; else → replanner.
"""

import logging
from typing import Any, Dict

from chat_reasoner.state import (
    ALLOWED_TOOLS,
    ChatReasonerState,
    PlanValidationResult,
)

logger = logging.getLogger(__name__)

_VALIDATOR_RETRY_CAP = 3


def _has_cycle(steps: list) -> bool:
    """DFS cycle detection on the depends_on DAG."""
    id_to_deps = {s["step_id"]: set(s.get("depends_on", [])) for s in steps}
    visited = set()
    in_stack = set()

    def dfs(node):
        visited.add(node)
        in_stack.add(node)
        for dep in id_to_deps.get(node, set()):
            if dep not in visited:
                if dfs(dep):
                    return True
            elif dep in in_stack:
                return True
        in_stack.discard(node)
        return False

    for sid in id_to_deps:
        if sid not in visited:
            if dfs(sid):
                return True
    return False


def _validate_plan(plan: list) -> PlanValidationResult:
    failed = []

    # Check 1: at least one step
    if not plan:
        return PlanValidationResult(
            valid=False,
            failed_checks=["at_least_one_step"],
            feedback="الخطة فارغة. يجب أن تحتوي على خطوة واحدة على الأقل.",
        )

    step_ids = [s.get("step_id", "") for s in plan]

    # Check 2: unique step_ids
    if len(step_ids) != len(set(step_ids)):
        failed.append("unique_step_ids")

    # Check 3: valid tool names
    bad_tools = [s.get("tool") for s in plan if s.get("tool") not in ALLOWED_TOOLS]
    if bad_tools:
        failed.append("valid_tool_names")

    # Check 4: non-empty queries
    empty_queries = [s.get("step_id") for s in plan if not str(s.get("query", "")).strip()]
    if empty_queries:
        failed.append("non_empty_queries")

    # Check 5: depends_on references valid ids
    id_set = set(step_ids)
    bad_deps = []
    for s in plan:
        for dep in s.get("depends_on", []):
            if dep not in id_set:
                bad_deps.append(dep)
    if bad_deps:
        failed.append("depends_on_resolvable")

    # Check 6: acyclic
    if not failed and _has_cycle(plan):
        failed.append("acyclic")

    if not failed:
        return PlanValidationResult(valid=True, failed_checks=[], feedback="")

    # Build human-readable feedback
    msgs = []
    if "unique_step_ids" in failed:
        msgs.append("step_id مكرر — تأكد من أن كل step_id فريد.")
    if "valid_tool_names" in failed:
        msgs.append(f"أسماء أدوات غير صحيحة: {bad_tools}. الأدوات المسموحة: {sorted(ALLOWED_TOOLS)}.")
    if "non_empty_queries" in failed:
        msgs.append(f"حقل query فارغ في الخطوات: {empty_queries}.")
    if "depends_on_resolvable" in failed:
        msgs.append(f"depends_on يشير إلى step_id غير موجود: {bad_deps}.")
    if "acyclic" in failed:
        msgs.append("الخطة تحتوي على حلقة دائرية في depends_on.")

    return PlanValidationResult(
        valid=False,
        failed_checks=failed,
        feedback=" | ".join(msgs),
    )


def plan_validator_node(state: ChatReasonerState) -> Dict[str, Any]:
    plan = state.get("plan", [])
    result = _validate_plan(plan)
    logger.info(
        "Plan validator: valid=%s checks=%s", result.valid, result.failed_checks
    )

    if result.valid:
        return {
            "plan_validation_status": "valid",
            "plan_validation_feedback": "",
        }

    return {
        "plan_validation_status": "invalid",
        "plan_validation_feedback": result.feedback,
        "validator_retry_count": state.get("validator_retry_count", 0) + 1,
    }


def validator_router(state: ChatReasonerState) -> str:
    if state.get("plan_validation_status") == "valid":
        return "executor_fanout"
    retry = state.get("validator_retry_count", 0)
    if retry < _VALIDATOR_RETRY_CAP:
        return "planner"
    logger.warning("Validator retry cap reached; forcing replanner.")
    return "replanner"
