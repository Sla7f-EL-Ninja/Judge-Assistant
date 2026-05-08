"""
test_files_streaming_and_groups.py

Integration tests for:
  - GET /api/v1/files/{file_id}  (streaming endpoint)
  - Multi-file group ingestion   (IngestRequest.groups)
  - Legacy file_ids ingestion
  - IngestRequest validation
  - OCR endpoint on grouped document
  - Delete grouped document keeps files intact

All tests use real MongoDB, real MinIO, and real DocumentProcessor.
Tests skip cleanly if the required infrastructure is unavailable.

Marker: integration
"""

import io
from contextlib import asynccontextmanager
from typing import List, Optional

import pytest
from bson import ObjectId

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Small in-process fixtures
# ---------------------------------------------------------------------------

def _png_bytes() -> bytes:
    """Return a small valid PNG. Uses Pillow if available, else a hardcoded literal."""
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Minimal 1×1 red PNG — valid enough for upload + MIME validation
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
            b"\x00\xce\xcd\xc1\xb3\x00\x00\x00\x00IEND\xaeB`\x82"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.conftest import auth_headers


async def _upload(app_client, content: bytes, filename: str, mime: str) -> str:
    """Upload a file and return its file_id."""
    r = await app_client.post(
        "/api/v1/files/upload",
        files={"file": (filename, content, mime)},
        headers=auth_headers(),
    )
    assert r.status_code == 201, f"Upload failed ({r.status_code}): {r.text}"
    return r.json()["file_id"]


async def _create_case(app_client) -> str:
    """Create a test case and return its ID."""
    r = await app_client.post(
        "/api/v1/cases",
        json={"title": "integration-test-case"},
        headers=auth_headers(),
    )
    assert r.status_code == 201, f"Case creation failed ({r.status_code}): {r.text}"
    data = r.json()
    # CaseResponse serialises id with alias _id
    return data.get("_id") or data.get("id")


# ---------------------------------------------------------------------------
# Scratch context manager — tracks and cleans up created resources
# ---------------------------------------------------------------------------

class _Scratch:
    def __init__(self, app_client, motor_db):
        self._client = app_client
        self._db = motor_db
        self.file_ids: List[str] = []
        self.case_ids: List[str] = []
        self.doc_ids: List[str] = []

    async def cleanup(self):
        for case_id in self.case_ids:
            for doc_id in self.doc_ids:
                try:
                    await self._client.delete(
                        f"/api/v1/cases/{case_id}/documents/{doc_id}",
                        headers=auth_headers(),
                    )
                except Exception:
                    pass
            try:
                await self._client.delete(
                    f"/api/v1/cases/{case_id}",
                    headers=auth_headers(),
                )
            except Exception:
                pass
        for file_id in self.file_ids:
            try:
                await self._client.delete(
                    f"/api/v1/files/{file_id}",
                    headers=auth_headers(),
                )
            except Exception:
                pass


@asynccontextmanager
async def _scratch(app_client, motor_db):
    s = _Scratch(app_client, motor_db)
    try:
        yield s
    finally:
        await s.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFileStreaming:
    """GET /api/v1/files/{file_id} — raw file streaming."""

    async def test_get_file_streams_pdf(self, app_client, motor_client, test_pdf_bytes):
        async with _scratch(app_client, motor_client) as s:
            file_id = await _upload(app_client, test_pdf_bytes, "test.pdf", "application/pdf")
            s.file_ids.append(file_id)

            r = await app_client.get(f"/api/v1/files/{file_id}", headers=auth_headers())
            assert r.status_code == 200, r.text
            assert "application/pdf" in r.headers.get("content-type", ""), r.headers
            disposition = r.headers.get("content-disposition", "")
            assert disposition.startswith("inline;"), f"Expected inline, got: {disposition}"
            assert "test.pdf" in disposition
            assert len(r.content) == len(test_pdf_bytes)

    async def test_get_file_streams_png(self, app_client, motor_client):
        png = _png_bytes()
        async with _scratch(app_client, motor_client) as s:
            file_id = await _upload(app_client, png, "test.png", "image/png")
            s.file_ids.append(file_id)

            r = await app_client.get(f"/api/v1/files/{file_id}", headers=auth_headers())
            assert r.status_code == 200, r.text
            assert "image/png" in r.headers.get("content-type", ""), r.headers
            disposition = r.headers.get("content-disposition", "")
            assert disposition.startswith("inline;"), disposition
            assert len(r.content) == len(png)

    async def test_get_file_404_for_unknown_id(self, app_client, motor_client):
        r = await app_client.get(
            "/api/v1/files/file_doesnotexist_xyz999",
            headers=auth_headers(),
        )
        assert r.status_code == 404, r.text
        # Error shape: {"error": {"code": "...", "detail": "...", "status": N}}
        error = r.json().get("error", {})
        assert error.get("code") == "FILE_NOT_FOUND", r.json()

    async def test_get_file_download_query_flips_disposition(self, app_client, motor_client, test_pdf_bytes):
        async with _scratch(app_client, motor_client) as s:
            file_id = await _upload(app_client, test_pdf_bytes, "test.pdf", "application/pdf")
            s.file_ids.append(file_id)

            r = await app_client.get(
                f"/api/v1/files/{file_id}",
                params={"download": "1"},
                headers=auth_headers(),
            )
            assert r.status_code == 200, r.text
            disposition = r.headers.get("content-disposition", "")
            assert disposition.startswith("attachment;"), f"Expected attachment, got: {disposition}"


@pytest.mark.integration
class TestGroupIngestion:
    """POST /cases/{case_id}/documents with groups= or file_ids=."""

    async def test_ingest_groups_creates_single_document(
        self, app_client, motor_client, qdrant_client, test_pdf_bytes
    ):
        from api.db.collections import DOCUMENTS
        from config.supervisor import QDRANT_COLLECTION_CASE
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        async with _scratch(app_client, motor_client) as s:
            case_id = await _create_case(app_client)
            s.case_ids.append(case_id)

            fid_a = await _upload(app_client, test_pdf_bytes, "page1.pdf", "application/pdf")
            fid_b = await _upload(app_client, test_pdf_bytes, "page2.pdf", "application/pdf")
            s.file_ids.extend([fid_a, fid_b])

            r = await app_client.post(
                f"/api/v1/cases/{case_id}/documents",
                json={"groups": [{"file_ids": [fid_a, fid_b]}]},
                headers=auth_headers(),
            )
            assert r.status_code == 201, r.text
            body = r.json()
            assert len(body["ingested"]) == 1, body
            assert len(body["errors"]) == 0, body

            item = body["ingested"][0]
            assert item["file_ids"] == [fid_a, fid_b], item
            doc_id = item["doc_id"]
            assert doc_id, "doc_id must be populated"
            s.doc_ids.append(doc_id)

            # MongoDB: exactly one document row
            try:
                oid = ObjectId(doc_id)
            except Exception:
                oid = doc_id
            doc = await motor_client[DOCUMENTS].find_one({"_id": oid, "case_id": case_id})
            assert doc is not None, f"Document {doc_id} not found in MongoDB"
            assert doc.get("file_ids") == [fid_a, fid_b], doc.get("file_ids")
            assert len(doc.get("source_files", [])) == 2, doc.get("source_files")
            assert "--- PAGE BREAK ---" in (doc.get("text") or ""), "merged text must contain PAGE BREAK"

            # Qdrant: all chunks belong to same mongo_id
            scroll_result, _ = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_CASE,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="metadata.mongo_id",
                        match=MatchValue(value=doc_id),
                    )]
                ),
                limit=100,
                with_payload=True,
            )
            assert len(scroll_result) > 0, "No Qdrant chunks found for grouped doc"
            for point in scroll_result:
                assert point.payload.get("metadata", {}).get("mongo_id") == doc_id

    async def test_ingest_legacy_file_ids_creates_n_documents(
        self, app_client, motor_client, test_pdf_bytes
    ):
        from api.db.collections import DOCUMENTS
        from bson import ObjectId

        async with _scratch(app_client, motor_client) as s:
            case_id = await _create_case(app_client)
            s.case_ids.append(case_id)

            fid_a = await _upload(app_client, test_pdf_bytes, "doc1.pdf", "application/pdf")
            fid_b = await _upload(app_client, test_pdf_bytes, "doc2.pdf", "application/pdf")
            s.file_ids.extend([fid_a, fid_b])

            r = await app_client.post(
                f"/api/v1/cases/{case_id}/documents",
                json={"file_ids": [fid_a, fid_b]},
                headers=auth_headers(),
            )
            assert r.status_code == 201, r.text
            body = r.json()
            assert len(body["ingested"]) == 2, body
            assert len(body["errors"]) == 0, body

            doc_ids = [item["doc_id"] for item in body["ingested"] if item.get("doc_id")]
            s.doc_ids.extend(doc_ids)
            assert len(set(doc_ids)) == 2, f"Expected 2 distinct doc_ids, got {doc_ids}"


@pytest.mark.integration
class TestIngestRequestValidation:
    """IngestRequest schema validation rules."""

    async def test_rejects_both_forms(self, app_client, motor_client, test_pdf_bytes):
        async with _scratch(app_client, motor_client) as s:
            case_id = await _create_case(app_client)
            s.case_ids.append(case_id)

            fid = await _upload(app_client, test_pdf_bytes, "x.pdf", "application/pdf")
            s.file_ids.append(fid)

            r = await app_client.post(
                f"/api/v1/cases/{case_id}/documents",
                json={"file_ids": [fid], "groups": [{"file_ids": [fid]}]},
                headers=auth_headers(),
            )
            assert r.status_code == 422, r.text

    async def test_rejects_neither_form(self, app_client, motor_client):
        async with _scratch(app_client, motor_client) as s:
            case_id = await _create_case(app_client)
            s.case_ids.append(case_id)

            r = await app_client.post(
                f"/api/v1/cases/{case_id}/documents",
                json={},
                headers=auth_headers(),
            )
            assert r.status_code == 422, r.text


@pytest.mark.integration
class TestOCREndpointOnGroupedDoc:
    """GET /cases/{case_id}/documents/{doc_id}/ocr on a multi-file doc."""

    async def test_ocr_returns_merged_text_and_file_ids(
        self, app_client, motor_client, test_pdf_bytes
    ):
        async with _scratch(app_client, motor_client) as s:
            case_id = await _create_case(app_client)
            s.case_ids.append(case_id)

            fid_a = await _upload(app_client, test_pdf_bytes, "p1.pdf", "application/pdf")
            fid_b = await _upload(app_client, test_pdf_bytes, "p2.pdf", "application/pdf")
            s.file_ids.extend([fid_a, fid_b])

            ingest_r = await app_client.post(
                f"/api/v1/cases/{case_id}/documents",
                json={"groups": [{"file_ids": [fid_a, fid_b]}]},
                headers=auth_headers(),
            )
            assert ingest_r.status_code == 201, ingest_r.text
            doc_id = ingest_r.json()["ingested"][0]["doc_id"]
            s.doc_ids.append(doc_id)

            ocr_r = await app_client.get(
                f"/api/v1/cases/{case_id}/documents/{doc_id}/ocr",
                headers=auth_headers(),
            )
            assert ocr_r.status_code == 200, ocr_r.text
            ocr_data = ocr_r.json()

            assert "--- PAGE BREAK ---" in ocr_data.get("text", ""), "merged text missing PAGE BREAK"
            assert ocr_data.get("file_ids") == [fid_a, fid_b], ocr_data.get("file_ids")
            assert ocr_data.get("file_id") == fid_a, "deprecated alias must equal file_ids[0]"


@pytest.mark.integration
class TestDeleteGroupedDocument:
    """DELETE removes doc + Qdrant chunks, leaves file rows intact."""

    async def test_delete_grouped_doc_keeps_files(
        self, app_client, motor_client, qdrant_client, test_pdf_bytes
    ):
        from api.db.collections import DOCUMENTS, FILES
        from config.supervisor import QDRANT_COLLECTION_CASE
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        async with _scratch(app_client, motor_client) as s:
            case_id = await _create_case(app_client)
            s.case_ids.append(case_id)

            fid_a = await _upload(app_client, test_pdf_bytes, "del1.pdf", "application/pdf")
            fid_b = await _upload(app_client, test_pdf_bytes, "del2.pdf", "application/pdf")
            # Register files for cleanup (delete endpoint won't reach them after doc delete)
            s.file_ids.extend([fid_a, fid_b])

            ingest_r = await app_client.post(
                f"/api/v1/cases/{case_id}/documents",
                json={"groups": [{"file_ids": [fid_a, fid_b]}]},
                headers=auth_headers(),
            )
            assert ingest_r.status_code == 201, ingest_r.text
            doc_id = ingest_r.json()["ingested"][0]["doc_id"]
            # Do NOT register doc_id in s.doc_ids — the test itself deletes it

            del_r = await app_client.delete(
                f"/api/v1/cases/{case_id}/documents/{doc_id}",
                headers=auth_headers(),
            )
            assert del_r.status_code == 200, del_r.text

            # MongoDB: document row gone
            try:
                oid = ObjectId(doc_id)
            except Exception:
                oid = doc_id
            doc = await motor_client[DOCUMENTS].find_one({"_id": oid})
            assert doc is None, f"Document {doc_id} should be deleted from MongoDB"

            # Qdrant: no chunks remain
            scroll_result, _ = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_CASE,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="metadata.mongo_id",
                        match=MatchValue(value=doc_id),
                    )]
                ),
                limit=10,
            )
            assert len(scroll_result) == 0, f"Qdrant chunks still present for deleted doc {doc_id}"

            # MongoDB files: both file rows still present
            for fid in (fid_a, fid_b):
                file_rec = await motor_client[FILES].find_one({"_id": fid})
                assert file_rec is not None, f"File {fid} was unexpectedly deleted from MongoDB"
