"""
planner.py

Planner node: decomposes the judge's query into a validated step DAG.
Uses get_llm("high") with structured output bound to the Plan schema.
"""

import json
import logging
from typing import Any, Dict

from config import get_llm

from chat_reasoner.prompts import (
    PLANNER_CONTEXT_TEMPLATE,
    PLANNER_FEEDBACK_ADDENDUM,
    PLANNER_SYSTEM,
)
from chat_reasoner.state import ChatReasonerState, Plan
from chat_reasoner.state import ALLOWED_TOOLS  # ← state module

logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 8


def _format_history(conversation_history: list, n: int = _MAX_HISTORY_TURNS) -> str:
    turns = conversation_history[-n:] if conversation_history else []
    if not turns:
        return "لا يوجد سجل محادثة."
    lines = []
    for t in turns:
        role = "القاضي" if t.get("role") == "user" else "المساعد"
        lines.append(f"**{role}:** {t.get('content', '')}")
    return "\n".join(lines)


def planner_node(state: ChatReasonerState) -> Dict[str, Any]:
    logger.info("Planner: building step plan (validator_retry=%d)", state.get("validator_retry_count", 0))

    history_text = _format_history(state.get("conversation_history", []))

    context_block = PLANNER_CONTEXT_TEMPLATE.format(
        escalation_reason=state.get("escalation_reason", "غير محدد"),
        original_query=state["original_query"],
        n_turns=_MAX_HISTORY_TURNS,
        conversation_history=history_text,
    )

    if state.get("validator_retry_count", 0) > 0 and state.get("plan_validation_feedback"):
        context_block += PLANNER_FEEDBACK_ADDENDUM.format(
            plan_validation_feedback=state["plan_validation_feedback"]
        )

    prompt = PLANNER_SYSTEM.format(context_block=context_block)

    llm = get_llm("high").with_structured_output(Plan)

    try:
        plan: Plan = llm.invoke(prompt)

        # Coerce any near-miss tool names before validation
        for step in plan.steps:
            if step.tool not in ALLOWED_TOOLS:
                normalized = step.tool.lower().replace("-", "_").replace(" ", "_")
                match = next(
                    (t for t in ALLOWED_TOOLS if
                     t in normalized or normalized in t or
                     normalized.startswith(t[:6]) or t.startswith(normalized[:6])),
                    None,
                )
                if match:
                    logger.warning("Planner tool name coerced: %r → %r", step.tool, match)
                    step.tool = match
        logger.info("Planner: produced %d steps", len(plan.steps))
        return {
            "plan": [s.model_dump() for s in plan.steps],
            "plan_validation_status": "pending",
        }
    except Exception as exc:
        logger.exception("Planner LLM error: %s", exc)
        # Return an empty plan so the validator fails it and routes to replanner
        return {
            "plan": [],
            "plan_validation_status": "pending",
            "plan_validation_feedback": f"فشل LLM المخطط: {exc}",
        }
