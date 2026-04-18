"""
Summerize/pipeline.py
---------------------
Public programmatic interface for the summarization pipeline.

Any module in the project can do:

    from Summerize.pipeline import run_summarization

    result = run_summarization(
        documents=[{"doc_id": "file.txt", "raw_text": "..."}],
        case_id="abc123",           # optional — saves to MongoDB when given
        save_to_db=True,            # default True when case_id is provided
    )

    print(result.rendered_brief)
    print(result.all_sources)

The CLI (main.py) and the FastAPI endpoint are both thin wrappers
around this single entry point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import sys
from typing import List, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
from config import cfg, get_llm
from Summerize.graph import create_pipeline

logger = logging.getLogger("hakim.pipeline")

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class SummarizationResult:
    """Structured result returned by :func:`run_summarization`."""

    rendered_brief: str
    """Full brief as Arabic markdown — the primary human-readable output."""

    case_brief: dict
    """Structured 7-section dict (CaseBrief model fields)."""

    all_sources: List[str]
    """Unique citation strings collected across the brief."""

    role_theme_summaries: List[dict] = field(default_factory=list)
    themed_roles: List[dict]         = field(default_factory=list)
    role_aggregations: List[dict]    = field(default_factory=list)
    bullets_count: int               = 0
    chunks_count: int                = 0
    documents_count: int             = 0

    # Set by the pipeline after a successful MongoDB upsert
    saved_to_db: bool   = False
    case_id: Optional[str] = None


# ---------------------------------------------------------------------------
# MongoDB helper (sync — safe from CLI and thread-pool contexts)
# ---------------------------------------------------------------------------
 
def _upsert_summary(
    case_id: str,
    result: SummarizationResult,
) -> bool:
    """Upsert *result* into MongoDB under *case_id*.

    All connection settings come from ``cfg`` (settings.yaml / env vars).
    Returns True on success, False on failure so callers can react.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.error(
            "pymongo is not installed — cannot save to MongoDB. "
            "Run: pip install pymongo"
        )
        return False

    mongo_uri  = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
    db_name    = cfg.mongodb.get("database", "Rag")
    collection = "summaries"
    timeout_ms = cfg.mongodb.get("server_selection_timeout_ms", 5000)

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        doc = {
            "case_id":      case_id,
            "summary":      result.rendered_brief,
            "sources":      result.all_sources,
            "case_brief":   result.case_brief,
            "generated_at": datetime.now(timezone.utc),
        }
        client[db_name][collection].update_one(
            {"case_id": case_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info(
            "Summary upserted  db='%s'  collection='%s'  case_id='%s'",
            db_name, collection, case_id,
        )
        return True
    except Exception as exc:
        logger.error("MongoDB upsert failed: %s", exc)
        return False
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def run_summarization(
    documents: List[dict],
    *,
    case_id: Optional[str] = None,
    save_to_db: bool = True,
    llm=None,
) -> SummarizationResult:
    """Run the full Nodes 0-5 summarization pipeline and return a result object.

    Parameters
    ----------
    documents:
        List of dicts with keys ``"doc_id"`` (str) and ``"raw_text"`` (str).
        At least one non-empty document is required.
    case_id:
        When provided, the result is upserted into MongoDB under this key.
        Pass ``save_to_db=False`` to skip saving even when case_id is set.
    save_to_db:
        Controls the MongoDB upsert.  Defaults to True; only relevant when
        ``case_id`` is also provided.
    llm:
        Optional pre-built LangChain chat model.  When omitted the pipeline
        creates one via ``get_llm("high")`` from the central config.

    Returns
    -------
    SummarizationResult
        Always returned — even when the pipeline produces an empty brief —
        so callers never have to handle None.

    Raises
    ------
    ValueError
        If *documents* is empty or contains no non-empty raw_text entries.
    RuntimeError
        If the pipeline itself raises an unrecoverable error.
    """
    # ---- Validate input ----------------------------------------------------
    if not documents:
        raise ValueError("run_summarization: 'documents' must not be empty.")

    valid_docs = [d for d in documents if d.get("raw_text", "").strip()]
    if not valid_docs:
        raise ValueError("run_summarization: all documents have empty 'raw_text'.")

    # ---- Build pipeline ----------------------------------------------------
    if llm is None:
        llm = get_llm("high")

    app = create_pipeline(llm)

    initial_state = {
        "documents":            valid_docs,
        "chunks":               [],
        "classified_chunks":    [],
        "bullets":              [],
        "role_aggregations":    [],
        "themed_roles":         [],
        "role_theme_summaries": [],
        "case_brief":           {},
        "all_sources":          [],
        "rendered_brief":       "",
        "party_manifest":       {},
    }

    # ---- Run ---------------------------------------------------------------
    logger.info(
        "run_summarization: starting  docs=%d  case_id=%s",
        len(valid_docs), case_id,
    )
    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        raise RuntimeError(f"Summarization pipeline failed: {exc}") from exc

    # ---- Build result ------------------------------------------------------
    result = SummarizationResult(
        rendered_brief       = final_state.get("rendered_brief", ""),
        case_brief           = final_state.get("case_brief", {}),
        all_sources          = final_state.get("all_sources", []),
        role_theme_summaries = final_state.get("role_theme_summaries", []),
        themed_roles         = final_state.get("themed_roles", []),
        role_aggregations    = final_state.get("role_aggregations", []),
        bullets_count        = len(final_state.get("bullets", [])),
        chunks_count         = len(final_state.get("chunks", [])),
        documents_count      = len(valid_docs),
        case_id              = case_id,
    )

    logger.info(
        "run_summarization: done  sources=%d  brief_len=%d",
        len(result.all_sources), len(result.rendered_brief),
    )

    # ---- Persist -----------------------------------------------------------
    if case_id and save_to_db:
        result.saved_to_db = _upsert_summary(case_id, result)

    return result
