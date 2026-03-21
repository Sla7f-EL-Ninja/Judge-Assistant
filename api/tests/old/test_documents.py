"""
test_documents.py

Tests for POST /api/v1/cases/{case_id}/documents
"""

import asyncio
from datetime import datetime, timezone

from api.tests.conftest import TEST_USER_ID


def _seed_case(fake_db, case_id="case_doc"):
    """Insert a case into the fake DB."""
    now = datetime.now(timezone.utc)
    doc = {
        "_id": case_id,
        "user_id": TEST_USER_ID,
        "title": "Doc Case",
        "description": "",
        "status": "active",
        "metadata": {},
        "documents": [],
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    asyncio.run(fake_db["cases"].insert_one(doc))


def test_ingest_case_not_found(client):
    """Ingesting into a non-existent case should return 404."""
    resp = client.post(
        "/api/v1/cases/nonexistent/documents",
        json={"file_ids": ["file_1"]},
    )
    assert resp.status_code == 404


def test_ingest_file_not_found(client, fake_db):
    """Ingesting a non-existent file_id should report an error."""
    _seed_case(fake_db, case_id="case_doc")

    resp = client.post(
        "/api/v1/cases/case_doc/documents",
        json={"file_ids": ["file_nonexistent"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["errors"]) == 1
    assert data["errors"][0]["file_id"] == "file_nonexistent"
    assert "not found" in data["errors"][0]["error"].lower()


def test_ingest_empty_file_ids(client):
    """Sending an empty file_ids array should return 422."""
    resp = client.post(
        "/api/v1/cases/some_case/documents",
        json={"file_ids": []},
    )
    assert resp.status_code == 422
