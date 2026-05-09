"""
corpus_router.py
----------------
corpus_classifier_node: determines which legal corpus the query belongs to.

Runs AFTER preprocessor_node, which has already:
  - run the fast off-topic gate (non-Arabic / too short)
  - classified the query as "analytical" | "textual" (or "off_topic")
  - rewritten the query into rewritten_question

This node is therefore only invoked when classification != "off_topic".  Its
sole job is to pick the best-matching corpus above the confidence threshold.

Threshold logic:
    winner confidence >= corpus_router_threshold  → set corpus_config, continue
    winner confidence <  corpus_router_threshold  → overwrite classification to "off_topic"
                                                    (unsupported domain, e.g. family law)

State keys read:
    rewritten_question  (preferred) | last_query  (fallback)
    classification, llm_call_count

State keys written:
    corpus_config          CorpusConfig for the winning corpus (or None)
    corpus_routing_scores  List[dict] raw scores from the LLM (observability)
    classification         Overwritten to "off_topic" only when no corpus matched

Routing (via corpus_classifier_router in routers.py):
    "textual_node"          → corpus found + classification == "textual"
    "scope_classifier_node" → corpus found + classification == "analytical"
    "off_topic_node"        → no corpus matched (below threshold or LLM error)

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
import threading
from typing import Optional

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from config.legal_rag import get_llm, MAX_LLM_CALLS
from config import cfg
from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.errors import CorpusRoutingError
from RAG.legal_rag.llm_utils import invoke_with_budget_and_timeout
from RAG.legal_rag.prompts import CORPUS_ROUTER_PROMPT
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None
_llm_lock = threading.Lock()


def _get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = get_llm("medium")
    return _llm


def _corpus_threshold() -> float:
    return float(
        cfg.get("rag", {}).get("legal", {}).get("corpus_router_threshold", 0.4)
    )


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


# ---------------------------------------------------------------------------
# Lazy corpus registry — imported only on first call, never at module load.
# ---------------------------------------------------------------------------

_registry: dict | None = None
_registry_lock = threading.Lock()


def _get_registry() -> dict:
    global _registry
    if _registry is not None:
        return _registry
    with _registry_lock:
        if _registry is not None:
            return _registry
        # Intentionally deferred — do NOT move to top level.
        from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS
        from RAG.legal_rag.evidence_rag.corpus import EVIDENCE_CORPUS
        from RAG.legal_rag.procedures_rag.corpus import PROCEDURES_CORPUS

        _registry = {
            "civil":      CIVIL_LAW_CORPUS,
            "evidence":   EVIDENCE_CORPUS,
            "procedures": PROCEDURES_CORPUS,
        }
    return _registry


@traceable(name="Corpus Classifier Node")
def corpus_router_node(state: dict) -> dict:
    """Classify the query into a supported legal corpus.

    Reads rewritten_question (preferred) or last_query.  Does NOT run a fast
    off-topic gate — that is the preprocessor's responsibility.
    """
    query = (
        state.get("rewritten_question")
        or state.get("last_query")
        or ""
    ).strip()

    # ── LLM scoring ──────────────────────────────────────────────────────
    prompt = CORPUS_ROUTER_PROMPT.format(question=query)

    try:
        response = invoke_with_budget_and_timeout(
            state, _get_llm(), prompt, node="corpus_classifier_node"
        )
        data   = json.loads(_strip_fences(response.content.strip()))
        scores: list = data.get("scores", [])
    except Exception as exc:
        log_event(
            logger, "corpus_classifier_error", error=str(exc), level=logging.ERROR
        )
        state["corpus_routing_scores"] = []
        state["error"] = {
            "type":    CorpusRoutingError.__name__,
            "node":    "corpus_classifier_node",
            "message": str(exc),
        }
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
        # No supported corpus matched — mark as off_topic (unsupported domain).
        state["classification"] = "off_topic"
        log_event(
            logger, "corpus_classifier_no_match", scores=scores, threshold=threshold
        )
        return state

    corpus_name   = winner["corpus_name"]
    corpus_config = _get_registry().get(corpus_name)

    if corpus_config is None:
        # LLM returned an unknown corpus key — treat as off_topic.
        state["classification"] = "off_topic"
        log_event(
            logger, "corpus_classifier_unknown_corpus",
            corpus_name=corpus_name, level=logging.WARNING
        )
        return state

    state["corpus_config"] = corpus_config
    log_event(
        logger, "corpus_classifier",
        winner=corpus_name,
        confidence=winner.get("confidence"),
        reason=winner.get("reason", ""),
        scores=scores,
    )
    return state
