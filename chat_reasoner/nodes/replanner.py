"""
replanner.py

Replanner node: produces a revised plan when a step hits 3 consecutive failures
or when the Synthesizer flags insufficient context.

Unified path for both triggers (I4 improvement):
- Failure trigger: replan_trigger_step_id set, replan_trigger_error set.
- Synthesizer trigger: replan_trigger_step_id=None, replan_trigger_error="synthesizer_insufficient".

After producing a new plan, resets validator state and routes back to plan_validator.
Hard guard: if replan_count >= 2, marks status=failed and routes to trace_writer.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from config import get_llm

from chat_reasoner.prompts import (
    REPLANNER_CONTEXT_TEMPLATE,
    REPLANNER_SYSTEM,
)
from chat_reasoner.state import ChatReasonerState, Plan, _STEP_RESULTS_RESET

logger = logging.getLogger(__name__)

_MAX_REPLANS = 2
_MAX_HISTORY_TURNS = 8

def _ensure_dict(r):
    if isinstance(r, str):
        try:
            return __import__('json').loads(r)
        except Exception:
            return {}
    return r


def _format_history(conversation_history: list, n: int = _MAX_HISTORY_TURNS) -> str:
    turns = conversation_history[-n:] if conversation_history else []
    if not turns:
        return "لا يوجد سجل محادثة."
    lines = []
    for t in turns:
        role = "القاضي" if t.get("role") == "user" else "المساعد"
        lines.append(f"**{role}:** {t.get('content', '')}")
    return "\n".join(lines)


def _format_failed_plan(plan: list) -> str:
    if not plan:
        return "لا توجد خطة سابقة."
    lines = []
    for s in plan:
        deps = s.get("depends_on", [])
        dep_str = ", ".join(deps) if deps else "—"
        lines.append(
            f"- **{s['step_id']}** | tool: `{s.get('tool')}` | depends_on: {dep_str}\n"
            f"  query: {s.get('query', '')}"
        )
    return "\n".join(lines)


def _build_replan_reason(state: ChatReasonerState) -> str:
    trigger_error = state.get("replan_trigger_error", "")
    trigger_step = state.get("replan_trigger_step_id")

    if trigger_error == "synthesizer_insufficient":
        return f"المُجمِّع أشار إلى نقص في السياق: {state.get('final_answer', '')[:300]}"
    if trigger_step:
        return f"الخطوة `{trigger_step}` فشلت 3 مرات متتالية. آخر خطأ: {trigger_error}"
    return trigger_error or "سبب غير محدد"


def replanner_node(state: ChatReasonerState) -> Dict[str, Any]:
    replan_count = state.get("replan_count", 0)

    # Hard guard
    if replan_count >= _MAX_REPLANS:
        logger.error("Replanner: replan_count=%d >= cap — marking failed", replan_count)
        return {
            "status": "failed",
            "error_message": "تعذّر إكمال الاستدلال: استُنفدت محاولات إعادة التخطيط.",
        }

    logger.info("Replanner: attempt %d/%d", replan_count + 1, _MAX_REPLANS)

    replan_reason = _build_replan_reason(state)
    history_text = _format_history(state.get("conversation_history", []))
    failed_plan_text = _format_failed_plan(state.get("plan", []))

    context_block = REPLANNER_CONTEXT_TEMPLATE.format(
        escalation_reason=state.get("escalation_reason", "غير محدد"),
        original_query=state["original_query"],
        n_turns=_MAX_HISTORY_TURNS,
        conversation_history=history_text,
        failed_plan=failed_plan_text,
        replan_reason=replan_reason,
    )

    prompt = REPLANNER_SYSTEM.format(context_block=context_block)
    llm = get_llm("high").with_structured_output(Plan)

    try:
        new_plan: Plan = llm.invoke(prompt)
    except Exception as exc:
        logger.exception("Replanner LLM error: %s", exc)
        return {
            "status": "failed",
            "error_message": f"فشل LLM في إعادة التخطيط: {exc}",
        }

    new_plan_steps = [s.model_dump() for s in new_plan.steps]
    new_step_ids = {s["step_id"] for s in new_plan_steps}

    # Prune step_failures and step_results for orphaned step_ids
    pruned_failures = {
        k: v for k, v in state.get("step_failures", {}).items() if k in new_step_ids
    }
    pruned_results: List[dict] = []  # empty → reducer resets step_results

    replan_event = {
        "replan_index": replan_count + 1,
        "trigger_step_id": state.get("replan_trigger_step_id"),
        "trigger_error": state.get("replan_trigger_error"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "new_plan_step_ids": list(new_step_ids),
    }

    logger.info(
        "Replanner: produced %d new steps — %s",
        len(new_plan_steps),
        list(new_step_ids),
    )

    return {
        "plan": new_plan_steps,
        "plan_validation_status": "pending",
        "plan_validation_feedback": "",
        "validator_retry_count": 0,
        "step_failures": pruned_failures,
        "step_results": pruned_results,
        "replan_count": replan_count + 1,
        "replan_trigger_step_id": None,
        "replan_trigger_error": None,
        "replan_events": [replan_event],
    }