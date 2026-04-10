"""
tests/CASE_RAG/test_ingestion_quality.py

Layer A: Verify ingestion correctness.
All assertions are existence/shape-only (no LLM calls).
Depends on the `ingestion_results` session fixture (hard-fail if broken).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue

from conftest import EXPECTED_DOC_TYPES, FIXTURE_FILES, TEST_CASE_ID


# ---------------------------------------------------------------------------
# A1 -- all files ingested
# ---------------------------------------------------------------------------

def test_all_files_ingested(ingestion_results):
    """7 results returned, every mongo_id is non-None."""
    assert len(ingestion_results) == 7

    for r in ingestion_results:
        assert r.get("mongo_id") is not None, (
            f"mongo_id is None for: {r.get('file')}"
        )


# ---------------------------------------------------------------------------
# A2 -- classification accuracy
# ---------------------------------------------------------------------------

def test_classification_accuracy(ingestion_results):
    """Each file's doc_type matches the ground-truth inventory."""
    by_filename: dict[str, str] = {
        Path(r["file"]).name: r["doc_type"]
        for r in ingestion_results
    }

    for fname, expected_type in EXPECTED_DOC_TYPES.items():
        actual_type = by_filename.get(fname)
        assert actual_type == expected_type, (
            f"{fname}: expected doc_type='{expected_type}', got '{actual_type}'"
        )


# ---------------------------------------------------------------------------
# A3 -- confidence above threshold
# ---------------------------------------------------------------------------

def test_confidence_above_threshold(ingestion_results):
    """Every ingested document has confidence >= 50."""
    for r in ingestion_results:
        confidence = r.get("confidence", 0)
        assert confidence >= 50, (
            f"Low confidence ({confidence}) for: {r.get('file')}"
        )


# ---------------------------------------------------------------------------
# A4 -- MongoDB schema
# ---------------------------------------------------------------------------

def test_mongodb_schema(mongo_collection):
    """MongoDB contains exactly 7 docs for this case_id with required fields."""
    docs = list(mongo_collection.find({"case_id": TEST_CASE_ID}))

    assert len(docs) == 7, (
        f"Expected 7 MongoDB docs, found {len(docs)}"
    )

    required_fields = ("title", "doc_type", "case_id", "text", "source_file")
    for doc in docs:
        for field in required_fields:
            assert field in doc, (
                f"Field '{field}' missing in MongoDB doc: {doc.get('title')}"
            )
            value = doc[field]
            assert value is not None and str(value).strip() != "", (
                f"Field '{field}' is empty in MongoDB doc: {doc.get('title')}"
            )


# ---------------------------------------------------------------------------
# A5 -- Qdrant vectors exist
# ---------------------------------------------------------------------------

def test_qdrant_vectors_exist(file_ingestor):
    """Qdrant collection has >= 7 points tagged with this case_id.

    Each point must carry the required metadata keys.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant_client = file_ingestor.vectorstore.client
    collection_name = file_ingestor._qdrant_collection_name

    scroll_result, _ = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.case_id",
                    match=MatchValue(value=TEST_CASE_ID),
                )
            ]
        ),
        limit=200,
        with_payload=True,
    )

    assert len(scroll_result) >= 7, (
        f"Expected >= 7 Qdrant points for case_id={TEST_CASE_ID}, "
        f"found {len(scroll_result)}"
    )

    required_metadata_keys = (
        "case_id", "title", "type", "source_file", "mongo_id", "chunk_index"
    )
    for point in scroll_result:
        payload = point.payload or {}
        metadata = payload.get("metadata", {})
        for key in required_metadata_keys:
            assert key in metadata, (
                f"Qdrant point {point.id} missing metadata key '{key}'. "
                f"Available keys: {list(metadata.keys())}"
            )


# ---------------------------------------------------------------------------
# A6 -- titles match doc_types (FileIngestor convention)
# ---------------------------------------------------------------------------

def test_titles_match_doc_types(ingestion_results):
    """Every ingestion result has title == doc_type (FileIngestor convention)."""
    for r in ingestion_results:
        title = r.get("title", "")
        doc_type = r.get("doc_type", "")
        assert title == doc_type, (
            f"title '{title}' != doc_type '{doc_type}' for: {r.get('file')}"
        )
