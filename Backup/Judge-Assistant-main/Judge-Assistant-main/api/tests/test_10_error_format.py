"""
test_error_format.py

Dedicated tests that verify every error response from the API matches the
standard ErrorEnvelope schema:

    {
        "error": {
            "code": "<ERROR_CODE>",
            "detail": "<human-readable message>",
            "status": <http_status_code>
        }
    }
"""

import pytest
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient
from jose import jwt

from config.api import get_settings
from conftest import auth_headers


HEADERS = auth_headers()


def _assert_error_envelope(response, expected_status: int, expected_code: str | None = None):
    """Assert that a response matches the ErrorEnvelope schema."""
    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "error" in body, f"Missing 'error' key in response: {body}"
    err = body["error"]
    assert "code" in err, f"Missing 'code' in error object: {err}"
    assert "detail" in err, f"Missing 'detail' in error object: {err}"
    assert "status" in err, f"Missing 'status' in error object: {err}"
    assert err["status"] == response.status_code, (
        f"error.status ({err['status']}) does not match HTTP status ({response.status_code})"
    )
    if expected_code is not None:
        assert err["code"] == expected_code, (
            f"Expected error code '{expected_code}', got '{err['code']}'"
        )


# ---------------------------------------------------------------------------
# 401: Authentication errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_401_missing_auth_header(client: AsyncClient):
    """Missing Authorization header returns 422 with ErrorEnvelope."""
    r = await client.get("/api/v1/cases")
    _assert_error_envelope(r, 422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_error_401_invalid_bearer_prefix(client: AsyncClient):
    """Authorization header without 'Bearer ' prefix returns 401."""
    r = await client.get(
        "/api/v1/cases", headers={"Authorization": "Token abc123"}
    )
    _assert_error_envelope(r, 401, "UNAUTHORIZED")


@pytest.mark.asyncio
async def test_error_401_tampered_token(client: AsyncClient):
    """Tampered JWT token returns 401 with ErrorEnvelope."""
    s = get_settings()
    good = jwt.encode(
        {"user_id": "u1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        s.jwt_secret,
        algorithm=s.jwt_algorithm,
    )
    tampered = good[:-1] + ("X" if good[-1] != "X" else "Y")
    r = await client.get(
        "/api/v1/cases", headers={"Authorization": f"Bearer {tampered}"}
    )
    _assert_error_envelope(r, 401, "UNAUTHORIZED")


@pytest.mark.asyncio
async def test_error_401_expired_token(client: AsyncClient):
    """Expired JWT token returns 401 with ErrorEnvelope."""
    s = get_settings()
    token = jwt.encode(
        {"user_id": "u1", "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
        s.jwt_secret,
        algorithm=s.jwt_algorithm,
    )
    r = await client.get(
        "/api/v1/cases", headers={"Authorization": f"Bearer {token}"}
    )
    _assert_error_envelope(r, 401, "UNAUTHORIZED")


# ---------------------------------------------------------------------------
# 404: Not found errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_404_nonexistent_case(client: AsyncClient):
    """Requesting a case that does not exist returns 404 with CASE_NOT_FOUND."""
    r = await client.get(
        "/api/v1/cases/case_doesnotexist_999", headers=HEADERS
    )
    _assert_error_envelope(r, 404, "CASE_NOT_FOUND")


@pytest.mark.asyncio
async def test_error_404_nonexistent_conversation(client: AsyncClient):
    """Requesting a conversation that does not exist returns 404."""
    r = await client.get(
        "/api/v1/conversations/conv_doesnotexist_999", headers=HEADERS
    )
    _assert_error_envelope(r, 404, "CONVERSATION_NOT_FOUND")


@pytest.mark.asyncio
async def test_error_404_nonexistent_summary(client: AsyncClient):
    """Requesting a summary for a nonexistent case returns 404."""
    r = await client.get(
        "/api/v1/cases/case_doesnotexist_999/summary", headers=HEADERS
    )
    _assert_error_envelope(r, 404, "CASE_NOT_FOUND")


# ---------------------------------------------------------------------------
# 400: Bad request errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_400_invalid_mime_type(client: AsyncClient):
    """Uploading a file with disallowed MIME type returns 400."""
    r = await client.post(
        "/api/v1/files/upload",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 400, "INVALID_MIME_TYPE")


@pytest.mark.asyncio
async def test_error_400_oversized_file(client: AsyncClient):
    """Uploading an oversized file returns 400."""
    oversized = b"A" * (20 * 1024 * 1024 + 1)
    r = await client.post(
        "/api/v1/files/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 400, "FILE_TOO_LARGE")


@pytest.mark.asyncio
async def test_error_400_empty_patch_body(client: AsyncClient):
    """Sending a PATCH with no updatable fields returns 400."""
    r = await client.patch(
        "/api/v1/cases/case_doesnotexist_999",
        json={},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 400, "NO_FIELDS_TO_UPDATE")


# ---------------------------------------------------------------------------
# 422: Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_error_422_empty_case_title(client: AsyncClient):
    """Creating a case with empty title returns 422 with VALIDATION_ERROR."""
    r = await client.post(
        "/api/v1/cases",
        json={"title": ""},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_error_422_invalid_case_status(client: AsyncClient):
    """Updating a case with an invalid status value returns 422."""
    r = await client.patch(
        "/api/v1/cases/some_case_id",
        json={"status": "banana"},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_error_422_invalid_limit_value(client: AsyncClient):
    """Passing limit=999 (above max 100) returns 422."""
    r = await client.get("/api/v1/cases?limit=999", headers=HEADERS)
    _assert_error_envelope(r, 422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_error_422_empty_query_string(client: AsyncClient):
    """Submitting a query with empty string returns 422."""
    r = await client.post(
        "/api/v1/query",
        json={"query": "", "case_id": "some_case"},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_error_422_missing_required_fields(client: AsyncClient):
    """Submitting a case create with missing title field returns 422."""
    r = await client.post(
        "/api/v1/cases",
        json={},
        headers=HEADERS,
    )
    _assert_error_envelope(r, 422, "VALIDATION_ERROR")
