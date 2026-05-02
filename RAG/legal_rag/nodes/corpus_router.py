# """
# corpus_router.py
# ----------------
# Corpus Router node: decides which legal corpus the query belongs to
# BEFORE any query rewriting or type classification occurs.

# Uses CORPUS_ROUTER_PROMPT (already in prompts.py) to score the query
# against each available corpus, then picks the highest-confidence one.

# Threshold logic:
#     - winner confidence >= CORPUS_CONFIDENCE_THRESHOLD  → set corpus_config, continue
#     - winner confidence <  CORPUS_CONFIDENCE_THRESHOLD  → mark off_topic

# State keys written:
#     corpus_config          CorpusConfig for the winning corpus (or None)
#     corpus_routing_scores  List[dict] raw scores from the LLM (observability)
#     classification         Set to "off_topic" if no corpus clears threshold

# Routing (via corpus_router_router in routers.py):
#     "preprocessor_node"  → corpus found, proceed normally
#     "off_topic_node"     → no corpus matched (below threshold or LLM error)
# """

# from __future__ import annotations

# import json
# import logging
# import re
# from typing import Optional

# from dotenv import load_dotenv
# from langsmith import traceable

# load_dotenv()

# from RAG.legal_rag.config import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT, cfg
# from RAG.legal_rag.corpus_config import CorpusConfig
# from RAG.legal_rag.prompts import CORPUS_ROUTER_PROMPT
# from RAG.legal_rag.telemetry import get_logger, log_event

# logger = get_logger(__name__)

# # ---------------------------------------------------------------------------
# # Corpus registry — single source of truth for the unified graph.
# # Import each corpus's singleton CorpusConfig here.
# # ---------------------------------------------------------------------------
# from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS
# from RAG.legal_rag.evidence_rag.corpus import EVIDENCE_CORPUS
# from RAG.legal_rag.procedures_rag.corpus import PROCEDURES_CORPUS

# _CORPUS_REGISTRY: dict[str, CorpusConfig] = {
#     "civil":      CIVIL_LAW_CORPUS,
#     "evidence":   EVIDENCE_CORPUS,
#     "procedures": PROCEDURES_CORPUS,
# }

# _llm = None


# def _get_llm():
#     global _llm
#     if _llm is None:
#         _llm = get_llm("medium")
#     return _llm


# def _corpus_threshold() -> float:
#     return float(
#         cfg.get("rag", {}).get("legal", {}).get("corpus_router_threshold", 0.4)
#     )


# def _strip_fences(text: str) -> str:
#     return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


# def _fast_off_topic(query: str) -> bool:
#     """Cheap pre-check before spending an LLM call."""
#     q = query.strip()
#     if len(q) < 5:
#         return True
#     if not re.search(r"[\u0600-\u06FF]", q):
#         return True
#     return False


# @traceable(name="Corpus Router Node")
# def corpus_router_node(state: dict) -> dict:
#     """Score the query against all corpora and inject the winning CorpusConfig."""
#     query = state.get("last_query", "").strip()

#     # ── Fast off-topic gate ───────────────────────────────────────────────
#     if _fast_off_topic(query):
#         state["classification"]        = "off_topic"
#         state["corpus_routing_scores"] = []
#         log_event(logger, "corpus_router_fast_off_topic", query=query[:100])
#         return state

#     # ── LLM budget guard ─────────────────────────────────────────────────
#     if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
#         # Graceful degradation: default to civil law so the pipeline can
#         # still attempt an answer rather than hard-failing.
#         state["corpus_config"]         = _CORPUS_REGISTRY["civil"]
#         state["corpus_routing_scores"] = []
#         log_event(logger, "corpus_router_budget_fallback",
#                   fallback="civil", level=logging.WARNING)
#         return state

#     # ── LLM scoring ──────────────────────────────────────────────────────
#     prompt = CORPUS_ROUTER_PROMPT.format(question=query)

#     try:
#         response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
#         state["llm_call_count"] = state.get("llm_call_count", 0) + 1
#         data = json.loads(_strip_fences(response.content.strip()))
#         scores: list[dict] = data.get("scores", [])
#     except Exception as exc:
#         log_event(logger, "corpus_router_error", error=str(exc),
#                   level=logging.WARNING)
#         # Fallback: default to civil law rather than crashing the pipeline.
#         state["corpus_config"]         = _CORPUS_REGISTRY["civil"]
#         state["corpus_routing_scores"] = []
#         return state

#     state["corpus_routing_scores"] = scores

#     # ── Pick winner ───────────────────────────────────────────────────────
#     threshold = _corpus_threshold()
#     winner: Optional[dict] = None

#     for entry in scores:
#         if entry.get("confidence", 0) < threshold:
#             continue
#         if winner is None or entry["confidence"] > winner["confidence"]:
#             winner = entry

#     if winner is None:
#         state["classification"] = "off_topic"
#         log_event(logger, "corpus_router_no_match",
#                   scores=scores, threshold=threshold)
#         return state

#     corpus_name = winner["corpus_name"]
#     corpus_config = _CORPUS_REGISTRY.get(corpus_name)

#     if corpus_config is None:
#         # Unknown corpus name returned by LLM — treat as off_topic.
#         state["classification"] = "off_topic"
#         log_event(logger, "corpus_router_unknown_corpus",
#                   corpus_name=corpus_name, level=logging.WARNING)
#         return state

#     state["corpus_config"] = corpus_config

#     log_event(logger, "corpus_router",
#               winner=corpus_name,
#               confidence=winner.get("confidence"),
#               reason=winner.get("reason", ""),
#               scores=scores)
#     return state


"""
corpus_router.py
----------------
Corpus Router node: decides which legal corpus the query belongs to
BEFORE any query rewriting or type classification occurs.

Uses CORPUS_ROUTER_PROMPT (already in prompts.py) to score the query
against each available corpus, then picks the highest-confidence one.

Threshold logic:
    - winner confidence >= CORPUS_CONFIDENCE_THRESHOLD  → set corpus_config, continue
    - winner confidence <  CORPUS_CONFIDENCE_THRESHOLD  → mark off_topic

State keys written:
    corpus_config          CorpusConfig for the winning corpus (or None)
    corpus_routing_scores  List[dict] raw scores from the LLM (observability)
    classification         Set to "off_topic" if no corpus clears threshold

Routing (via corpus_router_router in routers.py):
    "preprocessor_node"  → corpus found, proceed normally
    "off_topic_node"     → no corpus matched (below threshold or LLM error)

Note on imports:
    Corpus singletons are imported LAZILY inside _get_registry() to avoid
    a circular import.  The cycle was:
        graph.py → corpus_router.py → civil_law_rag/__init__.py → graph.py
    Lazy import breaks the cycle because corpus_router module-level code
    no longer touches graph.py's importers at load time.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from RAG.legal_rag.config import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT, cfg
from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.prompts import CORPUS_ROUTER_PROMPT
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("medium")
    return _llm


def _corpus_threshold() -> float:
    return float(
        cfg.get("rag", {}).get("legal", {}).get("corpus_router_threshold", 0.4)
    )


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _fast_off_topic(query: str) -> bool:
    """Cheap pre-check before spending an LLM call."""
    q = query.strip()
    if len(q) < 5:
        return True
    if not re.search(r"[\u0600-\u06FF]", q):
        return True
    return False


# ---------------------------------------------------------------------------
# Lazy corpus registry — imported only on first call, never at module load.
# This breaks the circular import:
#   graph.py → corpus_router.py → civil_law_rag/__init__.py → graph.py
# ---------------------------------------------------------------------------

_registry: dict | None = None


def _get_registry() -> dict:
    global _registry
    if _registry is not None:
        return _registry

    # These imports are intentionally deferred — do NOT move them to the top.
    from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS
    from RAG.legal_rag.evidence_rag.corpus import EVIDENCE_CORPUS
    from RAG.legal_rag.procedures_rag.corpus import PROCEDURES_CORPUS

    _registry = {
        "civil":      CIVIL_LAW_CORPUS,
        "evidence":   EVIDENCE_CORPUS,
        "procedures": PROCEDURES_CORPUS,
    }
    return _registry


@traceable(name="Corpus Router Node")
def corpus_router_node(state: dict) -> dict:
    """Score the query against all corpora and inject the winning CorpusConfig."""
    query = state.get("last_query", "").strip()

    # ── Fast off-topic gate ───────────────────────────────────────────────
    if _fast_off_topic(query):
        state["classification"]        = "off_topic"
        state["corpus_routing_scores"] = []
        log_event(logger, "corpus_router_fast_off_topic", query=query[:100])
        return state

    # ── LLM budget guard ─────────────────────────────────────────────────
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        state["corpus_config"]         = _get_registry()["civil"]
        state["corpus_routing_scores"] = []
        log_event(logger, "corpus_router_budget_fallback",
                  fallback="civil", level=logging.WARNING)
        return state

    # ── LLM scoring ──────────────────────────────────────────────────────
    prompt = CORPUS_ROUTER_PROMPT.format(question=query)

    try:
        response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
        state["llm_call_count"] = state.get("llm_call_count", 0) + 1
        data   = json.loads(_strip_fences(response.content.strip()))
        scores: list = data.get("scores", [])
    except Exception as exc:
        log_event(logger, "corpus_router_error", error=str(exc),
                  level=logging.WARNING)
        state["corpus_config"]         = _get_registry()["civil"]
        state["corpus_routing_scores"] = []
        return state

    state["corpus_routing_scores"] = scores

    # ── Pick winner ───────────────────────────────────────────────────────
    threshold = _corpus_threshold()
    winner: Optional[dict] = None

    for entry in scores:
        if entry.get("confidence", 0) < threshold:
            continue
        if winner is None or entry["confidence"] > winner["confidence"]:
            winner = entry

    if winner is None:
        state["classification"] = "off_topic"
        log_event(logger, "corpus_router_no_match",
                  scores=scores, threshold=threshold)
        return state

    corpus_name   = winner["corpus_name"]
    corpus_config = _get_registry().get(corpus_name)

    if corpus_config is None:
        state["classification"] = "off_topic"
        log_event(logger, "corpus_router_unknown_corpus",
                  corpus_name=corpus_name, level=logging.WARNING)
        return state

    state["corpus_config"] = corpus_config

    log_event(logger, "corpus_router",
              winner=corpus_name,
              confidence=winner.get("confidence"),
              reason=winner.get("reason", ""),
              scores=scores)
    return state