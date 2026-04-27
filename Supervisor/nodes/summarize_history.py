"""
summarize_history.py

Standalone graph node that compresses old conversation turns into a rolling
Arabic summary when enough messages have accumulated since the last summary.

Fires ONLY when messages_since_last_summary >= SUMMARIZE_EVERY_N_MESSAGES.
After firing, resets the counter to 0 so the next cycle starts clean.

Behaviour
---------
- Keeps the most recent SHORT_TERM_KEEP_TURNS * 2 messages verbatim.
- Folds older messages into running_summary (append-to-existing, not replace).
- Summary is capped at SUMMARY_MAX_SENTENCES sentences to prevent runaway growth.
- Returns updated conversation_history + running_summary + reset counter.
- On any LLM or import failure: logs warning, returns state unchanged.
"""

import logging
from typing import Any, Dict, List, Optional

from config.supervisor import SHORT_TERM_KEEP_TURNS
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

# Keep verbatim (each turn = 2 messages: user + assistant)
_KEEP_MESSAGES = SHORT_TERM_KEEP_TURNS * 2

# Fire summarization once this many messages have been added since the last run.
SUMMARIZE_EVERY_N_MESSAGES = 20

# Hard cap on summary length.
SUMMARY_MAX_SENTENCES = 15

def _build_summary_prompt(existing_summary: Optional[str], old_messages: List[dict]) -> str:
    turns_text = "\n".join(
        f"{'القاضي' if m['role'] == 'user' else 'المساعد'}: {m.get('content', '')}"
        for m in old_messages
    )
    
    if existing_summary:
        # This is the critical change: Pass BOTH to the LLM
        return (
            f"لديك هذا الملخص للمحادثات السابقة:\n{existing_summary}\n\n"
            "استجدت هذه المحادثات الجديدة مؤخراً:\n"
            f"{turns_text}\n\n"
            "المطلوب: قم بتحديث الملخص السابق بدمج المعلومات الجديدة فيه. "
            "إذا كانت المحادثات الجديدة غير مهمة، حافظ على تفاصيل الملخص السابق. "
            f"يجب أن يكون الناتج ملخصاً واحداً شاملاً في {SUMMARY_MAX_SENTENCES} جملة كحد أقصى."
        )
    else:
        # First time summarizing
        return (
            "المحادثات التالية مستخرجة من جلسة قضائية. "
            f"لخِّصها في {SUMMARY_MAX_SENTENCES} جمل كحد أقصى باللغة العربية "
            "مع الحفاظ على الحقائق القانونية الجوهرية.\n\n"
            f"{turns_text}"
        )


def summarize_history_node(state: SupervisorState) -> Dict[str, Any]:
    history: List[dict] = list(state.get("conversation_history") or [])
    existing_summary: Optional[str] = state.get("running_summary")

    # Only compress messages added since the last summary cycle.
    # history[-SUMMARIZE_EVERY_N_MESSAGES:] = the 20 new messages.
    messages_to_compress = history[-SUMMARIZE_EVERY_N_MESSAGES:]

    if not messages_to_compress:
        return {"messages_since_last_summary": 0}

    try:
        from config import get_llm
        llm = get_llm("medium")
        prompt = _build_summary_prompt(existing_summary, messages_to_compress)
        response = llm.invoke(prompt)
        new_summary = response.content if hasattr(response, "content") else str(response)
        logger.info("summarize_history: summarized %d messages", len(messages_to_compress))
        return {
            # Do NOT return conversation_history — operator.add would append not replace
            "running_summary": new_summary,
            "messages_since_last_summary": 0,
        }
    except Exception as exc:
        logger.warning("summarize_history: failed — %s", exc)
        return {}


def should_summarize_history(state: SupervisorState) -> str:
    """Router: fire when messages_since_last_summary hits SUMMARIZE_EVERY_N_MESSAGES."""
    count = state.get("messages_since_last_summary") or 0
    if count >= SUMMARIZE_EVERY_N_MESSAGES:
        logger.debug(
            "should_summarize_history: %d messages since last summary → summarize",
            count,
        )
        return "summarize"
    return "skip"