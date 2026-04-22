"""
classify_intent.py

Intent classification node for the Supervisor workflow.

Uses an LLM with structured output to classify the judge query into
one of the supported intents and determine which agents to invoke.
"""

import logging
import os
from typing import Any, Dict

from config import get_llm
from config.supervisor import VALID_INTENTS, AGENT_NAMES
from Supervisor.prompts import (
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_USER_TEMPLATE,
)
from Supervisor.state import IntentClassification, SupervisorState

logger = logging.getLogger(__name__)


def classify_intent_node(state: SupervisorState) -> Dict[str, Any]:
    """Analyse the judge query and decide which agent(s) to invoke.

    Updates state keys: ``intent``, ``target_agents``, ``classified_query``.
    """
    judge_query = state.get("judge_query", "")
    conversation_history = state.get("conversation_history", [])
    uploaded_files = state.get("uploaded_files", [])

    # Guard against empty query (B20)
    if not judge_query or not judge_query.strip():
        logger.warning("Empty judge_query received; routing to off_topic")
        return {
            "intent": "off_topic",
            "target_agents": [],
            "classified_query": "",
        }

    # Guard against oversized input (B35)
    MAX_QUERY_CHARS = 4000
    if len(judge_query) > MAX_QUERY_CHARS:
        logger.warning(
            "judge_query too long (%d chars > %d); truncating", len(judge_query), MAX_QUERY_CHARS
        )
        judge_query = judge_query[:MAX_QUERY_CHARS]

    # Format conversation history for the prompt
    history_text = ""
    if conversation_history:
        lines = []
        for turn in conversation_history[-10:]:  # last 10 turns for context
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")
        history_text = "\n".join(lines)
    else:
        history_text = "(لا يوجد سجل محادثة سابق)"

    # Only expose basenames — full paths may contain traversal components (G5.4.4)
    safe_filenames = [os.path.basename(f) for f in uploaded_files]
    uploaded_files_text = ", ".join(safe_filenames) if safe_filenames else "لا يوجد"

    user_prompt = INTENT_CLASSIFICATION_USER_TEMPLATE.format(
        conversation_history=history_text,
        judge_query=judge_query,
        uploaded_files=uploaded_files_text,
    )

    # Remind the LLM to populate every required field in IntentClassification
    user_prompt += (
        "\n\nيجب أن يحتوي ردك على الحقول التالية بالضبط:\n"
        "- intent: أحد القيم (civil_law_rag, case_doc_rag, reason, multi, off_topic)\n"
        "- target_agents: قائمة بأسماء العملاء المطلوب استدعاؤهم\n"
        "- rewritten_query: إعادة صياغة السؤال بشكل مستقل\n"
        "- reasoning: شرح موجز لقرار التصنيف\n"
    )

    try:
        llm = get_llm("medium")
        structured_llm = llm.with_structured_output(IntentClassification)

        messages = [
            {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result: IntentClassification = structured_llm.invoke(messages)

        if result is None:
            raise ValueError(
                "Structured output returned None — LLM did not populate all required fields"
            )

        # Validate and normalise the intent
        intent = result.intent.strip().lower()
        if intent not in VALID_INTENTS:
            logger.warning(
                "LLM returned unknown intent '%s', falling back to off_topic", intent
            )
            intent = "off_topic"

        # Normalise and deduplicate target_agents (G5.2.1)
        _seen: set = set()
        target_agents = []
        for a in result.target_agents:
            name = a.strip().lower()
            if name in AGENT_NAMES and name not in _seen:
                _seen.add(name)
                target_agents.append(name)

        # Enforce consistency between intent and target_agents (G5.2.2)
        if intent in AGENT_NAMES:
            if intent not in target_agents:
                # Declared intent missing from agents list — override
                target_agents = [intent]
            elif len(target_agents) > 1:
                # Single intent but LLM stuffed extra agents — cap to declared intent
                logger.warning(
                    "Single intent '%s' had extra agents %s; capping", intent, target_agents
                )
                target_agents = [intent]

        # If multi but no valid agents, fall back to off_topic
        if intent == "multi" and not target_agents:
            intent = "off_topic"
            target_agents = []

        if intent == "off_topic":
            target_agents = []

        # Enforce topological order so downstream agents see prior results (G5.2.3)
        # civil_law_rag → case_doc_rag → reason
        _AGENT_ORDER = {"civil_law_rag": 0, "case_doc_rag": 1, "reason": 2}
        if len(target_agents) > 1:
            target_agents = sorted(
                target_agents, key=lambda a: _AGENT_ORDER.get(a, 99)
            )

        logger.info(
            "Intent classified: %s -> agents=%s | reasoning: %s",
            intent, target_agents, result.reasoning,
        )

        return {
            "intent": intent,
            "target_agents": target_agents,
            "classified_query": result.rewritten_query or judge_query,
        }

    except Exception as exc:
        logger.exception("Intent classification failed: %s", exc)
        # Conservative fallback: treat as off-topic
        return {
            "intent": "off_topic",
            "target_agents": [],
            "classified_query": judge_query,
        }