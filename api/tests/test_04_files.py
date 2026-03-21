"""
test_files.py

Upload a real PDF and verify it lands on disk + in MongoDB.
Also tests the validation gates (wrong MIME type, oversized file).

The file_id is stored in TestState for the document ingestion test.
"""

import os
import pytest
from httpx import AsyncClient

from conftest import TestState, auth_headers

HEADERS = auth_headers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_pdf() -> bytes:
    """
    A tiny but valid single-page PDF (hand-crafted, no library needed).
    Used when TEST_PDF_PATH is not set, so upload tests still run.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    )


def _pdf_content(test_pdf_path) -> tuple[bytes, str]:
    """Return (bytes, filename) for upload — real file if provided, else minimal."""
    if test_pdf_path and os.path.isfile(test_pdf_path):
        with open(test_pdf_path, "rb") as f:
            return f.read(), os.path.basename(test_pdf_path)
    return _minimal_pdf(), "test_document.pdf"


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_pdf_returns_201(client: AsyncClient, state: TestState, test_pdf_path):
    content, filename = _pdf_content(test_pdf_path)
    r = await client.post(
        "/api/v1/files/upload",
        files={"file": (filename, content, "application/pdf")},
        headers=HEADERS,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "file_id" in body
    assert body["mime_type"] == "application/pdf"
    assert body["size_bytes"] == len(content)
    assert body["filename"] == filename
    state.file_id = body["file_id"]  # store for ingestion test


@pytest.mark.asyncio
async def test_uploaded_file_has_id_prefix(client: AsyncClient, state: TestState):
    assert state.file_id, "file_id not set — run test_upload_pdf_returns_201 first"
    assert state.file_id.startswith("file_"), (
        f"Expected file_id to start with 'file_', got: {state.file_id}"
    )


@pytest.mark.asyncio
async def test_upload_wrong_mime_type_rejected(client: AsyncClient):
    r = await client.post(
        "/api/v1/files/upload",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        headers=HEADERS,
    )
    assert r.status_code == 400
    assert "not allowed" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_oversized_file_rejected(client: AsyncClient):
    # Build a byte string just over the 20 MB limit
    oversized = b"A" * (20 * 1024 * 1024 + 1)
    r = await client.post(
        "/api/v1/files/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
        headers=HEADERS,
    )
    assert r.status_code == 400
    assert "exceeds" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_image_png_accepted(client: AsyncClient):
    """PNG is also an allowed MIME type — make sure it's accepted."""
    # 1x1 pixel valid PNG
    tiny_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    r = await client.post(
        "/api/v1/files/upload",
        files={"file": ("photo.png", tiny_png, "image/png")},
        headers=HEADERS,
    )
    assert r.status_code == 201, r.text
