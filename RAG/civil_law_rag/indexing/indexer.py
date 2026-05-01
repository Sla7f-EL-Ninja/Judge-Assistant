"""
indexer.py
----------
Civil Law offline indexing pipeline.

This module is ONLY for offline/startup use — it must NEVER be imported
at graph-construction time (avoids Qdrant network call on graph import).

The correct integration point is the FastAPI lifespan in api/app.py::

    from RAG.civil_law_rag.indexing.indexer import ensure_civil_law_indexed
    ensure_civil_law_indexed()

Can also be run as a standalone CLI::

    python -m RAG.civil_law_rag.indexing.indexer

Pipeline:
    1. Load tagged text file
    2. Split into structured Documents (tag-based parser)
    3. Batch embed + upsert into Qdrant
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader

from RAG.civil_law_rag.config import BATCH_SIZE, CORPUS_VERSION, DOCS_PATH
from RAG.civil_law_rag.indexing.splitter import split_egyptian_civil_law
from RAG.civil_law_rag.retrieval.vectorstore import (
    COLLECTION_NAME,
    get_qdrant_client,
    load_vectorstore,
)
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collection_point_count() -> int:
    """Return the number of points in the civil law Qdrant collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(COLLECTION_NAME)
        return info.points_count or 0
    except Exception as exc:
        is_not_found = (
            getattr(exc, "status_code", None) == 404
            or "not found" in str(exc).lower()
        )
        if is_not_found:
            return 0
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_civil_law(force: bool = False) -> None:
    """Run the full civil-law indexing pipeline.

    Args:
        force: If True, re-index even when the collection already has data.

    Raises:
        FileNotFoundError: if the source text file is missing.
    """
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Civil law source file not found: {DOCS_PATH}\n"
            "Place the tagged civil law file in RAG/civil_law_rag/docs/."
        )

    db = load_vectorstore()

    if not force and _collection_point_count() > 0:
        log_event(logger, "indexing_skipped", reason="already_populated",
                  collection=COLLECTION_NAME)
        return

    log_event(logger, "indexing_start", docs_path=DOCS_PATH,
              corpus_version=CORPUS_VERSION)

    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    document = loader.load()
    raw_text = document[0].page_content

    # No normalize() call here — the tag-based splitter handles
    # structure parsing directly from the tagged format
    docs = split_egyptian_civil_law(raw_text)

    log_event(logger, "indexing_parsed", total_docs=len(docs))

    if not docs:
        log_event(logger, "indexing_aborted", reason="no_documents_parsed",
                  docs_path=DOCS_PATH, level=logging.ERROR)
        return

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        db.add_documents(batch)
        log_event(logger, "indexing_progress",
                  indexed=i + len(batch), total=len(docs))

    log_event(logger, "indexing_complete", total_docs=len(docs),
              corpus_version=CORPUS_VERSION)


def ensure_civil_law_indexed() -> None:
    """Ensure the corpus is present in Qdrant. Fast no-op if already populated.

    This is the canonical startup entry point — call it from api/app.py
    lifespan, NOT from graph.py or any module imported at request time.
    """
    if _collection_point_count() > 0:
        log_event(logger, "indexing_check", status="already_present",
                  collection=COLLECTION_NAME)
        return

    log_event(logger, "indexing_check", status="empty_running_auto_index",
              collection=COLLECTION_NAME)
    index_civil_law()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    index_civil_law(force=force)
    print("Indexing complete.")
