"""
synthesizer.py

Synthesizer node: produces the final Arabic answer from all step results.
Uses get_llm("high") with SynthesizerDecision structured output.

Routing:
- sufficient=True  → trace_writer
- sufficient=False AND run_count < 2 → replanner (bump run_count)
- sufficient=False AND run_count >= 2 → trace_writer (best-effort)
"""

import logging
from typing import Any, Dict, List

from config import get_llm

from chat_reasoner.prompts import (
    STEP_RESULT_TEMPLATE,
    SYNTHESIZER_SYSTEM,
)
from chat_reasoner.state import (
    ChatReasonerState,
    SynthesizerDecision,
)

logger = logging.getLogger(__name__)

_MAX_RUN_COUNT = 2
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

def _ensure_dict(r):
    if isinstance(r, str):
        try:
            import json
            return json.loads(r)
        except Exception:
            return {}
    return r

def _format_step_results(step_results: List[dict]) -> str:
    if not step_results:
        return "لا توجد نتائج."
    blocks = []
    seen_ids = set()
    for r in [_ensure_dict(x) for x in step_results]:
        sid = r.get("step_id", "?")
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        status = r.get("status", "unknown")
        if status == "skipped":
            response_block = "*(الخطوة تُجوّزت — لا يوجد ملخص مُعدّ مسبقاً)*"
        elif status == "failure":
            response_block = f"*(فشلت الخطوة — {r.get('error', 'خطأ غير محدد')})*"
        else:
            response = r.get("response", "")
            sources = r.get("sources", [])
            response_block = response
            if sources:
                response_block += f"\n**المصادر:** {', '.join(sources)}"
        blocks.append(
            STEP_RESULT_TEMPLATE.format(
                step_id=sid,
                tool=r.get("tool", "?"),
                query=r.get("query", ""),
                status=status,
                response_block=response_block,
            )
        )
    return "\n\n".join(blocks)


def _dedupe_sources(step_results: List[dict]) -> List[str]:
    seen = set()
    out = []
    for r in step_results:
        for src in r.get("sources", []):
            normalized = src.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                out.append(src.strip())
    return out


def synthesizer_node(state: ChatReasonerState) -> Dict[str, Any]:
    logger.info(
        "Synthesizer: attempt (synthesis_attempts=%d run_count=%d)",
        state.get("synthesis_attempts", 0),
        state.get("run_count", 0),
    )

    step_results = state.get("step_results", [])
    history_text = _format_history(state.get("conversation_history", []))
    step_results_block = _format_step_results(step_results)

    prompt = SYNTHESIZER_SYSTEM.format(
        original_query=state["original_query"],
        escalation_reason=state.get("escalation_reason", "غير محدد"),
        n_turns=_MAX_HISTORY_TURNS,
        conversation_history=history_text,
        step_results_block=step_results_block,
    )

    llm = get_llm("high").with_structured_output(SynthesizerDecision)

    try:
        decision: SynthesizerDecision = llm.invoke(prompt)
    except Exception as exc:
        logger.exception("Synthesizer LLM error: %s", exc)
        return {
            "synthesis_attempts": state.get("synthesis_attempts", 0) + 1,
            "final_answer": "تعذّر توليد الإجابة النهائية بسبب خطأ في النموذج.",
            "final_sources": _dedupe_sources(step_results),
            "status": "succeeded",  # best-effort
        }

    final_sources = _dedupe_sources(step_results)

    logger.info("Synthesizer: sufficient=%s", decision.sufficient)

    updates: Dict[str, Any] = {
        "synthesis_attempts": state.get("synthesis_attempts", 0) + 1,
        "final_answer": decision.answer,
        "final_sources": final_sources,
        "synth_sufficient": decision.sufficient,
    }

    # If synthesizer wants a re-run, set up the replanner trigger here
    if not decision.sufficient and state.get("run_count", 0) < _MAX_RUN_COUNT:
        updates["run_count"] = state.get("run_count", 0) + 1
        updates["replan_trigger_step_id"] = None
        updates["replan_trigger_error"] = (
            f"synthesizer_insufficient: {decision.insufficiency_reason or 'context insufficient'}"
        )

    return updates


def synth_router(state: ChatReasonerState) -> str:
    sufficient = state.get("synth_sufficient", True)

    if sufficient:
        return "trace_writer"

    run_count = state.get("run_count", 0)
    if run_count < _MAX_RUN_COUNT:
        logger.info("Synthesizer insufficient; triggering replanner (run_count=%d)", run_count)
        return "replanner"

    logger.warning(
        "Synthesizer insufficient but run_count=%d >= cap; best-effort answer.", run_count
    )
    return "trace_writer"
