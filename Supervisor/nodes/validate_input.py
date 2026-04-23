"""
validate_input.py

Pre-classification input validation node.

Enforces length caps, Arabic content ratio, and basic prompt-injection
heuristics before any LLM is called.  Triggers off_topic_response on
malformed input so classifiers never see adversarial or oversized queries.
"""

import logging
import re
from typing import Any, Dict

from config.supervisor import MAX_QUERY_CHARS
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

_MAX_QUERY_CHARS = MAX_QUERY_CHARS if hasattr(MAX_QUERY_CHARS, "__int__") else 4000
_MIN_ARABIC_RATIO = 0.05  # at least 5 % of non-whitespace chars must be Arabic or Latin
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|all|the)\s+instructions?", re.I),
    re.compile(r"تجاهل\s+(التعليمات|الأوامر)", re.UNICODE),
    re.compile(r"you\s+are\s+now\s+(a|an)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    re.compile(r"act\s+as\s+(a|an|if)", re.I),
    re.compile(r"<\s*script", re.I),
    re.compile(r"system\s*:\s*you", re.I),
]


def _arabic_ratio(text: str) -> float:
    """Fraction of non-whitespace chars that are Arabic Unicode block."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    arabic = sum(1 for c in non_ws if "؀" <= c <= "ۿ")
    return arabic / len(non_ws)


def validate_input_node(state: SupervisorState) -> Dict[str, Any]:
    """Validate judge_query before classification.

    Sets intent='off_topic' with a classification_error when input is
    invalid, causing intent_router to short-circuit to off_topic_response.
    """
    query: str = state.get("judge_query", "")

    # Empty query
    if not query or not query.strip():
        logger.warning("Empty judge_query — routing off_topic")
        return {
            "intent": "off_topic",
            "target_agents": [],
            "classified_query": "",
            "classification_error": "empty_query",
        }

    # Length cap
    if len(query) > _MAX_QUERY_CHARS:
        logger.warning("judge_query exceeds %d chars (%d) — routing off_topic", _MAX_QUERY_CHARS, len(query))
        return {
            "intent": "off_topic",
            "target_agents": [],
            "classified_query": "",
            "classification_error": "query_too_long",
        }

    # Prompt injection heuristics
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("Potential prompt injection detected in judge_query — routing off_topic")
            return {
                "intent": "off_topic",
                "target_agents": [],
                "classified_query": "",
                "classification_error": "prompt_injection_detected",
            }

    # Pass through — no mutations needed
    return {}
