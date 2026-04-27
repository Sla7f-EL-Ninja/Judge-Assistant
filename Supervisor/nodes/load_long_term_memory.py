"""
load_long_term_memory.py

Graph node that runs after validate_input and before classify_intent.

Loads two long-term memory types from the MongoDB store:
  - semantic_facts   — factual case knowledge extracted from prior sessions
  - procedural_prefs — judge behavioral preferences learned over time

Both are injected into SupervisorState for downstream prompt assembly.
On any failure the node returns empty values and logs a warning — the graph
always continues regardless of store availability.
"""

import logging
from typing import Any, Dict, List, Optional

from config.supervisor import PROCEDURAL_INJECT_MAX_CHARS, SEMANTIC_FACTS_TOP_K
from Supervisor.services.memory import get_semantic_manager, get_store
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def load_long_term_memory_node(state: SupervisorState) -> Dict[str, Any]:
    """Populate semantic_facts and procedural_prefs from long-term store."""
    case_id: str = state.get("case_id", "")
    user_id: Optional[str] = state.get("user_id")
    judge_query: str = state.get("judge_query", "")

    semantic_facts: List[Dict[str, Any]] = []
    procedural_prefs: Optional[str] = None

    # -- Semantic facts (case-scoped) --
    if case_id:
        try:
            store = get_store()
            results = store.search(
                ("case", case_id, "facts"),
                query=judge_query,
                limit=SEMANTIC_FACTS_TOP_K,
            )
            semantic_facts = [r["value"] for r in results if isinstance(r.get("value"), dict)]
            logger.debug("load_long_term_memory: loaded %d semantic facts for case %s", len(semantic_facts), case_id)
        except Exception as exc:
            logger.warning("load_long_term_memory: semantic load failed — %s", exc)

    # -- Procedural prefs (user-scoped) --
    if user_id:
        try:
            store = get_store()
            results = store.search(
                ("user", user_id, "prefs"),
                limit=10,
            )
            if results:
                raw = "\n".join(
                    r["value"].get("content", "") if isinstance(r.get("value"), dict) else str(r.get("value", ""))
                    for r in results
                )
                procedural_prefs = raw[:PROCEDURAL_INJECT_MAX_CHARS] or None
            logger.debug("load_long_term_memory: loaded procedural prefs for user %s", user_id)
        except Exception as exc:
            logger.warning("load_long_term_memory: procedural load failed — %s", exc)

    return {
        "semantic_facts": semantic_facts,
        "procedural_prefs": procedural_prefs,
    }
