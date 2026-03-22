"""
test_cases.py

Full lifecycle for case management against real MongoDB:
  create -> list -> get -> update -> delete

The created case_id is stored in the shared TestState so later tests
(documents, query, conversations) can reference it.
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


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_case_returns_201(client: AsyncClient, state: TestState):
    r = await client.post(
        "/api/v1/cases",
        json={
            "title": "قضية اختبار تكامل النظام",
            "description": "قضية أُنشئت تلقائيًا بواسطة مجموعة الاختبار",
            "metadata": {"court": "محكمة الاستئناف", "year": 2024},
        },
        headers=HEADERS,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["title"] == "قضية اختبار تكامل النظام"
    assert body["status"] == "active"
    assert body["user_id"] == state.user_id
    assert "_id" in body
    state.case_id = body["_id"]  # store for later tests


@pytest.mark.asyncio
async def test_create_case_metadata_round_trip(client: AsyncClient, state: TestState):
    """Verify that metadata dict is persisted and returned correctly."""
    assert state.case_id
    r = await client.get(f"/api/v1/cases/{state.case_id}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["metadata"]["court"] == "محكمة الاستئناف"
    assert body["metadata"]["year"] == 2024


@pytest.mark.asyncio
async def test_create_case_has_documents_array(client: AsyncClient, state: TestState):
    """Verify that the response includes a documents array."""
    assert state.case_id
    r = await client.get(f"/api/v1/cases/{state.case_id}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body
    assert isinstance(body["documents"], list)


@pytest.mark.asyncio
async def test_create_case_conversation_count_is_int(client: AsyncClient, state: TestState):
    """Verify conversation_count is a valid integer."""
    assert state.case_id
    r = await client.get(f"/api/v1/cases/{state.case_id}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "conversation_count" in body
    assert isinstance(body["conversation_count"], int)
    assert body["conversation_count"] >= 0


@pytest.mark.asyncio
async def test_create_case_empty_title_rejected(client: AsyncClient):
    r = await client.post(
        "/api/v1/cases",
        json={"title": ""},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_cases_contains_created_case(client: AsyncClient, state: TestState):
    assert state.case_id, "case_id not set -- run test_create_case_returns_201 first"
    r = await client.get("/api/v1/cases", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "cases" in body and "total" in body
    ids = [c["_id"] for c in body["cases"]]
    assert state.case_id in ids, f"Created case {state.case_id} not found in list"


@pytest.mark.asyncio
async def test_list_cases_pagination(client: AsyncClient):
    r = await client.get("/api/v1/cases?skip=0&limit=1", headers=HEADERS)
    assert r.status_code == 200
    assert len(r.json()["cases"]) <= 1


@pytest.mark.asyncio
async def test_list_cases_invalid_limit_rejected(client: AsyncClient):
    r = await client.get("/api/v1/cases?limit=999", headers=HEADERS)
    _assert_error_envelope(r, 422)


# ---------------------------------------------------------------------------
# Get single
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_case_returns_correct_doc(client: AsyncClient, state: TestState):
    assert state.case_id
    r = await client.get(f"/api/v1/cases/{state.case_id}", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["_id"] == state.case_id
    assert "conversation_count" in body


@pytest.mark.asyncio
async def test_get_nonexistent_case_returns_404(client: AsyncClient):
    r = await client.get("/api/v1/cases/case_doesnotexist999", headers=HEADERS)
    _assert_error_envelope(r, 404)


@pytest.mark.asyncio
async def test_get_case_wrong_user_returns_404(client: AsyncClient, state: TestState):
    """Another user must not be able to see this case."""
    assert state.case_id
    other_headers = auth_headers("other_user_999")
    r = await client.get(f"/api/v1/cases/{state.case_id}", headers=other_headers)
    _assert_error_envelope(r, 404)


# ---------------------------------------------------------------------------
# Update (PATCH)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_case_title(client: AsyncClient, state: TestState):
    assert state.case_id
    r = await client.patch(
        f"/api/v1/cases/{state.case_id}",
        json={"title": "قضية اختبار -- عنوان محدث"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["title"] == "قضية اختبار -- عنوان محدث"


@pytest.mark.asyncio
async def test_update_case_status_to_archived(client: AsyncClient, state: TestState):
    assert state.case_id
    r = await client.patch(
        f"/api/v1/cases/{state.case_id}",
        json={"status": "archived"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "archived"


@pytest.mark.asyncio
async def test_update_case_invalid_status_rejected(client: AsyncClient, state: TestState):
    assert state.case_id
    r = await client.patch(
        f"/api/v1/cases/{state.case_id}",
        json={"status": "banana"},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422)


@pytest.mark.asyncio
async def test_update_case_empty_body_rejected(client: AsyncClient, state: TestState):
    assert state.case_id
    r = await client.patch(
        f"/api/v1/cases/{state.case_id}",
        json={},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 400)


# ---------------------------------------------------------------------------
# Restore to active for later tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_restore_case_to_active(client: AsyncClient, state: TestState):
    assert state.case_id
    r = await client.patch(
        f"/api/v1/cases/{state.case_id}",
        json={"status": "active"},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "active"
