"""
chat_reasoner_adapter.py

Adapter for the Chat Reasoner sub-agent.

Inherits from the local AgentAdapter (no Supervisor/ imports) so the
Supervisor dispatcher can duck-type invoke() and read .response/.sources/
.raw_output/.error — identical field contract as all other adapters.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from chat_reasoner.graph import app as chat_reasoner_app
from chat_reasoner.interface import AgentAdapter, AgentResult
from chat_reasoner.state import ChatReasonerState

logger = logging.getLogger(__name__)


class ChatReasonerAdapter(AgentAdapter):
    """Fallback multi-step reasoning sub-agent.

    Dispatched by the Supervisor when a query requires cross-tool decomposition:
    comparing legal articles, cross-referencing case documents against laws, or
    analyzing facts across multiple sources.

    Expected context keys:
        case_id (str)
        conversation_history (List[dict])
        escalation_reason (str)  — short note from intent classifier
    """

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            case_id = context.get("case_id", "")
            started_at = datetime.now(timezone.utc).isoformat()

            initial_state: ChatReasonerState = {
                # Input
                "original_query": query,
                "case_id": case_id,
                "conversation_history": context.get("conversation_history", []),
                "escalation_reason": context.get("escalation_reason", ""),
                # Planning
                "plan": [],
                "plan_validation_status": "pending",
                "plan_validation_feedback": "",
                "validator_retry_count": 0,
                # Execution
                "step_results": [],
                "step_failures": {},
                # Replan
                "replan_count": 0,
                "replan_trigger_step_id": None,
                "replan_trigger_error": None,
                # Synthesis
                "run_count": 0,
                "synthesis_attempts": 0,
                "final_answer": "",
                "final_sources": [],
                "synth_sufficient": True,
                # Control
                "status": "running",
                "error_message": None,
                # Trace
                "session_id": f"{case_id}::{started_at}",
                "started_at": started_at,
                "tool_calls_log": [],
                "replan_events": [],
            }

            result = chat_reasoner_app.invoke(initial_state)

            if result.get("status") == "failed":
                return AgentResult(
                    response="",
                    error=(
                        result.get("error_message")
                        or "تعذّر إكمال الاستدلال متعدد الخطوات."
                    ),
                )

            final_answer = result.get("final_answer", "")
            synth_sufficient = result.get("synth_sufficient", True)

            # Part 3.1: synth_sufficient=False means the synthesizer flagged the
            # answer as incomplete/partial.  Surface this instead of silently
            # delivering a truncated answer.
            if not synth_sufficient and final_answer:
                logger.warning(
                    "ChatReasoner: synth_sufficient=False — answer may be incomplete "
                    "(session_id=%s)", result.get("session_id", "")
                )
                final_answer += (
                    "\n\n---\n"
                    "**ملاحظة:** قد لا تكون هذه الإجابة شاملة — لم يتمكن نظام الاستدلال من "
                    "استيفاء جميع خطوات التحليل. يُنصح بالتحقق من النتائج."
                )

            return AgentResult(
                response=final_answer,
                sources=result.get("final_sources", []),
                raw_output={
                    "plan": result.get("plan", []),
                    "step_results": result.get("step_results", []),
                    "replan_count": result.get("replan_count", 0),
                    "run_count": result.get("run_count", 0),
                    "session_id": result.get("session_id", ""),
                    "synth_sufficient": synth_sufficient,
                },
            )

        except Exception as exc:
            logger.exception("ChatReasonerAdapter error: %s", exc)
            return AgentResult(
                response="",
                error=f"خطأ في وكيل الاستدلال الحواري: {exc}",
            )
