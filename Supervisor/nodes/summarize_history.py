"""
summarize_history.py

Standalone graph node that compresses old conversation turns into a rolling
Arabic summary when the history grows too large.

Fires ONLY when the should_summarize_history router decides the token count
exceeds SUMMARIZE_TRIGGER_TOKENS — never on every turn.

Behaviour
---------
- Keeps the most recent SHORT_TERM_KEEP_TURNS * 2 messages verbatim.
- Folds older messages into running_summary (append-to-existing, not replace).
- Returns updated conversation_history + running_summary.
- On any LLM or import failure: logs warning, returns state unchanged.
"""

import logging
from typing import Any, Dict, List, Optional

from config.supervisor import SHORT_TERM_KEEP_TURNS, SUMMARIZE_TRIGGER_TOKENS
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

# Keep verbatim (each turn = 2 messages: user + assistant)
_KEEP_MESSAGES = SHORT_TERM_KEEP_TURNS * 2


def _approx_tokens(messages: List[dict]) -> int:
    """Cheap token estimate: total chars / 4."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def _build_summary_prompt(existing_summary: Optional[str], old_messages: List[dict]) -> str:
    turns_text = "\n".join(
        f"{'القاضي' if m['role'] == 'user' else 'المساعد'}: {m.get('content', '')}"
        for m in old_messages
    )
    prefix = f"الملخص السابق:\n{existing_summary}\n\n" if existing_summary else ""
    return (
        f"{prefix}"
        "المحادثات التالية مستخرجة من جلسة قضائية. "
        "لخِّصها بدقة في فقرة أو اثنتين باللغة العربية مع الحفاظ على الحقائق القانونية والقضائية.\n\n"
        f"{turns_text}"
    )


def summarize_history_node(state: SupervisorState) -> Dict[str, Any]:
    """Compress old turns into running_summary; keep recent turns verbatim."""
    history: List[dict] = list(state.get("conversation_history") or [])
    existing_summary: Optional[str] = state.get("running_summary")

    if len(history) <= _KEEP_MESSAGES:
        # Nothing to compress — router should not have sent us here, but be safe.
        return {}

    old_messages = history[:-_KEEP_MESSAGES]
    recent_messages = history[-_KEEP_MESSAGES:]

    try:
        from config import get_llm  # noqa: PLC0415
        llm = get_llm("medium")
        prompt = _build_summary_prompt(existing_summary, old_messages)
        response = llm.invoke(prompt)
        new_summary = response.content if hasattr(response, "content") else str(response)
        logger.info(
            "summarize_history: compressed %d messages into summary (%d chars)",
            len(old_messages),
            len(new_summary),
        )
        return {
            "conversation_history": recent_messages,
            "running_summary": new_summary,
        }
    except Exception as exc:
        logger.warning("summarize_history: failed — %s; history unchanged", exc)
        return {}


def should_summarize_history(state: SupervisorState) -> str:
    """Router: return 'summarize' when history exceeds token threshold, else 'skip'."""
    history: List[dict] = state.get("conversation_history") or []
    tokens = _approx_tokens(history)
    if tokens > SUMMARIZE_TRIGGER_TOKENS and len(history) > _KEEP_MESSAGES:
        logger.debug("should_summarize_history: tokens=%d > threshold=%d → summarize", tokens, SUMMARIZE_TRIGGER_TOKENS)
        return "summarize"
    return "skip"
