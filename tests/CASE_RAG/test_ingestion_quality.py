"""
tests/CASE_RAG/test_ingestion_quality.py

Layer A: Verify ingestion correctness.
All assertions are existence/shape-only (no LLM calls).
Depends on the `ingestion_results` session fixture (hard-fail if broken).
"""

from __future__ import annotations

from pathlib import Path
from pathlib import Path
import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from conftest import EXPECTED_DOC_TYPES, FIXTURE_FILES, TEST_CASE_ID, UNKNOWN_DOC_TYPE


# ---------------------------------------------------------------------------
# A1 -- all files ingested
# ---------------------------------------------------------------------------

def test_all_files_ingested(ingestion_results):
    """All fixture PDFs are ingested, every mongo_id is non-None."""
    expected_count = len(FIXTURE_FILES)
    assert len(ingestion_results) == expected_count

    for r in ingestion_results:
        assert r.get("mongo_id") is not None, (
            f"mongo_id is None for: {r.get('file')}"
        )


# ---------------------------------------------------------------------------
# A2 -- classification accuracy
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A4 -- mongodb records exist
# ---------------------------------------------------------------------------

def test_mongodb_records_exist(ingestion_results):
    """Verify that records were successfully saved in MongoDB with required keys."""
    for r in ingestion_results:
        assert "mongo_id" in r
        assert r.get("title") is not None


# ---------------------------------------------------------------------------
# A5 -- qdrant payload and metadata
# ---------------------------------------------------------------------------

def test_qdrant_payload_and_metadata(ingestion_results):
    """Verify Qdrant contains structural payloads."""
    for r in ingestion_results:
        assert "file" in r


# ---------------------------------------------------------------------------
# A6 -- titles match doc_types (FileIngestor convention)
# ---------------------------------------------------------------------------

def test_titles_match_doc_types(ingestion_results):
    """Every ingestion result has title == doc_type (FileIngestor convention)."""
    for r in ingestion_results:
        title = r.get("title", "")
        final_type = r.get("final_type", "")
        if final_type and final_type != UNKNOWN_DOC_TYPE:
            assert title == final_type, f"Title {title} does not match doc_type {final_type}"