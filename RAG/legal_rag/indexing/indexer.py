"""
indexer.py
----------
Legal corpus offline indexing pipeline.

This module is ONLY for offline/startup use — it must NEVER be imported
at graph-construction time (avoids Qdrant network call on graph import).

The correct integration point is the FastAPI lifespan in api/app.py::

    from RAG.legal_rag.indexing.indexer import ensure_indexed
    from RAG.civil_law_rag.corpus import CIVIL_LAW_CORPUS
    from RAG.evidence_rag.corpus import EVIDENCE_CORPUS

    ensure_indexed(CIVIL_LAW_CORPUS)
    ensure_indexed(EVIDENCE_CORPUS)

Can also be run as a standalone CLI::

    python -m RAG.legal_rag.indexing.indexer --corpus civil_law
    python -m RAG.legal_rag.indexing.indexer --corpus evidence_law --force

Pipeline:
    1. Load tagged text file (path from CorpusConfig.docs_path)
    2. Split into structured Documents (tag-based parser)
    3. Batch embed + upsert into the corpus's Qdrant collection
"""

from __future__ import annotations

import logging
import os

from langchain_community.document_loaders import TextLoader

from config.legal_rag import BATCH_SIZE
from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.indexing.splitter import split_legal_document
from RAG.legal_rag.retrieval.vectorstore import (
    collection_point_count,
    load_vectorstore,
)
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_corpus(corpus_config: CorpusConfig, force: bool = False) -> None:
    """Run the full indexing pipeline for *corpus_config*.

    Args:
        corpus_config: Identifies which corpus to index (path, collection,
                       source value, version strings).
        force:         If True, re-index even when the collection already
                       has data.  Use after updating the source text file.

    Raises:
        FileNotFoundError: if corpus_config.docs_path is missing.
    """
    if not os.path.exists(corpus_config.docs_path):
        raise FileNotFoundError(
            f"Legal source file not found: {corpus_config.docs_path}\n"
            f"Expected corpus: {corpus_config.name}"
        )

    db = load_vectorstore(corpus_config.collection_name)

    if not force and collection_point_count(corpus_config.collection_name) > 0:
        log_event(logger, "indexing_skipped",
                  corpus=corpus_config.name,
                  reason="already_populated",
                  collection=corpus_config.collection_name)
        return

    log_event(logger, "indexing_start",
              corpus=corpus_config.name,
              docs_path=corpus_config.docs_path,
              corpus_version=corpus_config.corpus_version)

    loader   = TextLoader(corpus_config.docs_path, encoding="utf-8")
    document = loader.load()
    raw_text = document[0].page_content

    # No normalize() here — the tag-based splitter handles structure
    # parsing directly from the tagged format.
    docs = split_legal_document(raw_text, source_value=corpus_config.source_filter_value)

    log_event(logger, "indexing_parsed",
              corpus=corpus_config.name, total_docs=len(docs))

    if not docs:
        log_event(logger, "indexing_aborted",
                  corpus=corpus_config.name,
                  reason="no_documents_parsed",
                  docs_path=corpus_config.docs_path,
                  level=logging.ERROR)
        return

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        db.add_documents(batch)
        log_event(logger, "indexing_progress",
                  corpus=corpus_config.name,
                  indexed=i + len(batch),
                  total=len(docs))

    log_event(logger, "indexing_complete",
              corpus=corpus_config.name,
              total_docs=len(docs),
              corpus_version=corpus_config.corpus_version)


def ensure_indexed(corpus_config: CorpusConfig) -> None:
    """Ensure the corpus is present in Qdrant. Fast no-op if already populated.

    This is the canonical startup entry point — call it from api/app.py
    lifespan, NOT from graph.py or any module imported at request time.
    """
    if collection_point_count(corpus_config.collection_name) > 0:
        log_event(logger, "indexing_check",
                  corpus=corpus_config.name,
                  status="already_present",
                  collection=corpus_config.collection_name)
        return

    log_event(logger, "indexing_check",
              corpus=corpus_config.name,
              status="empty_running_auto_index",
              collection=corpus_config.collection_name)
    index_corpus(corpus_config)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(message)s")

    force   = "--force" in sys.argv
    corpus_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--corpus" and i + 1 < len(sys.argv):
            corpus_name = sys.argv[i + 1]

    if corpus_name == "civil_law":
        from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS as _cfg
    elif corpus_name == "evidence_law":
        from RAG.legal_rag.evidence_rag.corpus import EVIDENCE_CORPUS as _cfg
    else:
        print("Usage: python -m RAG.legal_rag.indexing.indexer --corpus <civil_law|evidence_law> [--force]")
        sys.exit(1)

    index_corpus(_cfg, force=force)
    print(f"Indexing complete: {_cfg.name}")
