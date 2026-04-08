"""
test_files.py

Tests for POST /api/v1/files/upload
"""

import io


def test_upload_pdf(client):
    """Uploading a valid PDF should return 201 with file metadata."""
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content")
    resp = client.post(
        "/api/v1/files/upload",
        files={"file": ("test.pdf", fake_pdf, "application/pdf")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == "test.pdf"
    assert data["mime_type"] == "application/pdf"
    assert data["size_bytes"] > 0
    assert "file_id" in data
    assert "uploaded_at" in data


def test_upload_image(client):
    """Uploading a valid PNG image should work."""
    fake_png = io.BytesIO(b"\x89PNG fake image data")
    resp = client.post(
        "/api/v1/files/upload",
        files={"file": ("scan.png", fake_png, "image/png")},
    )
    assert resp.status_code == 201
    assert resp.json()["mime_type"] == "image/png"


def test_upload_invalid_mime_type(client):
    """Uploading a disallowed MIME type should return 400."""
    fake_exe = io.BytesIO(b"MZ fake executable")
    resp = client.post(
        "/api/v1/files/upload",
        files={"file": ("malware.exe", fake_exe, "application/x-msdownload")},
    )
    assert resp.status_code == 400
    assert "MIME type" in resp.json()["detail"]


def test_upload_too_large(client, test_settings):
    """Uploading a file exceeding max size should return 400."""
    # Create content larger than the limit
    big_content = b"x" * (test_settings.max_upload_bytes + 1)
    resp = client.post(
        "/api/v1/files/upload",
        files={"file": ("big.pdf", io.BytesIO(big_content), "application/pdf")},
    )
    assert resp.status_code == 400
    assert "exceeds maximum" in resp.json()["detail"]


def test_upload_no_file(client):
    """Uploading without a file should return 422."""
    resp = client.post("/api/v1/files/upload")
    assert resp.status_code == 422
