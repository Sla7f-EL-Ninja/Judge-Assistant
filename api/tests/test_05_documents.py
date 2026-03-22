"""
test_documents.py

Ingest the uploaded file into the case through the real FileIngestor
(OCR -> classify -> embed -> store in Chroma).

Skipped automatically if:
  - TEST_PDF_PATH is not set (no real document to ingest)
  - file_id or case_id are not yet populated in TestState

These are the slowest tests -- ingestion can take 10-30 s per file.
"""

import pytest
from httpx import AsyncClient

from conftest import TestState, auth_headers

HEADERS = auth_headers()


def _assert_error_envelope(r, expected_status: int):
    """Verify response matches the standard ErrorEnvelope shape."""
    assert r.status_code == expected_status, r.text
    body = r.json()
    assert "error" in body, f"Missing 'error' key: {body}"
    err = body["error"]
    assert "code" in err and "detail" in err and "status" in err
    assert err["status"] == r.status_code


@pytest.mark.slow
@pytest.mark.asyncio
async def test_ingest_document_into_case(client: AsyncClient, state: TestState, test_pdf_path):
    if not test_pdf_path:
        pytest.skip("TEST_PDF_PATH not set -- skipping real ingestion test")
    if not state.file_id:
        pytest.skip("file_id not set -- run file upload tests first")
    if not state.case_id:
        pytest.skip("case_id not set -- run case tests first")

    r = await client.post(
        f"/api/v1/cases/{state.case_id}/documents",
        json={"file_ids": [state.file_id]},
        headers=HEADERS,
        timeout=120.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert "ingested" in body
    assert "errors" in body
    assert len(body["errors"]) == 0, (
        f"Ingestion errors: {body['errors']}"
    )
    assert len(body["ingested"]) == 1
    item = body["ingested"][0]
    assert item["file_id"] == state.file_id
    assert item["status"] == "success"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_case_documents_list_updated_after_ingest(client: AsyncClient, state: TestState, test_pdf_path):
    if not test_pdf_path:
        pytest.skip("TEST_PDF_PATH not set")
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.get(f"/api/v1/cases/{state.case_id}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    doc_ids = [d["file_id"] for d in body.get("documents", [])]
    assert state.file_id in doc_ids, (
        f"file_id {state.file_id} not found in case documents after ingestion.\n"
        f"Documents present: {doc_ids}"
    )


@pytest.mark.asyncio
async def test_ingest_nonexistent_file_returns_error(client: AsyncClient, state: TestState):
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        f"/api/v1/cases/{state.case_id}/documents",
        json={"file_ids": ["file_doesnotexist999"]},
        headers=HEADERS,
    )
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert len(body["errors"]) == 1
        assert "not found" in body["errors"][0]["error"].lower()


@pytest.mark.asyncio
async def test_ingest_into_nonexistent_case_returns_404(client: AsyncClient, state: TestState):
    if not state.file_id:
        pytest.skip("file_id not set")

    r = await client.post(
        "/api/v1/cases/case_doesnotexist999/documents",
        json={"file_ids": [state.file_id]},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 404)


@pytest.mark.asyncio
async def test_ingest_empty_file_ids_returns_422(client: AsyncClient, state: TestState):
    """Sending an empty file_ids list should return 422 validation error."""
    if not state.case_id:
        pytest.skip("case_id not set")

    r = await client.post(
        f"/api/v1/cases/{state.case_id}/documents",
        json={"file_ids": []},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422)
