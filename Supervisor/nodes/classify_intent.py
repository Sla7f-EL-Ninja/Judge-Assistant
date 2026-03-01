"""
classify_intent.py

Intent classification node for the Supervisor workflow.

Uses an LLM with structured output to classify the judge query into
one of the supported intents and determine which agents to invoke.
"""

import logging
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from Supervisor.config import LLM_MODEL, LLM_TEMPERATURE, VALID_INTENTS, AGENT_NAMES
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

    uploaded_files_text = ", ".join(uploaded_files) if uploaded_files else "لا يوجد"

    user_prompt = INTENT_CLASSIFICATION_USER_TEMPLATE.format(
        conversation_history=history_text,
        judge_query=judge_query,
        uploaded_files=uploaded_files_text,
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE
        )
        structured_llm = llm.with_structured_output(IntentClassification)

        messages = [
            {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result: IntentClassification = structured_llm.invoke(messages)

        # Validate and normalise the intent
        intent = result.intent.strip().lower()
        if intent not in VALID_INTENTS:
            logger.warning(
                "LLM returned unknown intent '%s', falling back to off_topic",
                intent,
            )
            intent = "off_topic"

        # Normalise target_agents
        target_agents = [
            a.strip().lower()
            for a in result.target_agents
            if a.strip().lower() in AGENT_NAMES
        ]

        # If single intent maps to a known agent, ensure it is in the list
        if intent in AGENT_NAMES and intent not in target_agents:
            target_agents = [intent]

        # If multi but no valid agents, fall back to off_topic
        if intent == "multi" and not target_agents:
            intent = "off_topic"
            target_agents = []

        if intent == "off_topic":
            target_agents = []

        logger.info(
            "Intent classified: %s -> agents=%s", intent, target_agents
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
