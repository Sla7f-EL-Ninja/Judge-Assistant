"""
test_summaries.py

Tests for GET /api/v1/cases/{case_id}/summary
"""

import asyncio
from datetime import datetime, timezone

from api.tests.conftest import TEST_USER_ID


def _seed_case_and_summary(fake_db, case_id="case_sum"):
    """Insert a case and its summary into the fake DB."""
    now = datetime.now(timezone.utc)

    case_doc = {
        "_id": case_id,
        "user_id": TEST_USER_ID,
        "title": "Summary Case",
        "description": "",
        "status": "active",
        "metadata": {},
        "documents": [],
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    summary_doc = {
        "case_id": case_id,
        "summary": "This is a structured summary of the case.",
        "generated_at": now.isoformat(),
        "sources": ["doc1.pdf", "doc2.pdf"],
    }

    asyncio.run(fake_db["cases"].insert_one(case_doc))
    asyncio.run(fake_db["summaries"].insert_one(summary_doc))


def test_get_summary(client, fake_db):
    """GET /api/v1/cases/{id}/summary should return the stored summary."""
    _seed_case_and_summary(fake_db, case_id="case_sum")

    resp = client.get("/api/v1/cases/case_sum/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["case_id"] == "case_sum"
    assert "structured summary" in data["summary"]
    assert len(data["sources"]) == 2


def test_get_summary_no_case(client):
    """Getting summary for a non-existent case should return 404."""
    resp = client.get("/api/v1/cases/nonexistent/summary")
    assert resp.status_code == 404
    assert "Case not found" in resp.json()["detail"]


def test_get_summary_no_summary(client, fake_db):
    """Getting summary for a case with no summary should return 404."""
    now = datetime.now(timezone.utc)
    case_doc = {
        "_id": "case_no_sum",
        "user_id": TEST_USER_ID,
        "title": "No Summary Case",
        "description": "",
        "status": "active",
        "metadata": {},
        "documents": [],
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    asyncio.run(fake_db["cases"].insert_one(case_doc))

    resp = client.get("/api/v1/cases/case_no_sum/summary")
    assert resp.status_code == 404
    assert "No summary" in resp.json()["detail"]
