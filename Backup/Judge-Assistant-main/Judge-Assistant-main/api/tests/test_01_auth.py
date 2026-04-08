"""
test_auth.py

Verify that the JWT guard works correctly:
  - Missing header   -> 422  (FastAPI required-header validation)
  - Wrong format     -> 401
  - Tampered token   -> 401
  - Expired token    -> 401
  - Valid token      -> passes through (tested implicitly by every other test)

All error responses are validated against the ErrorEnvelope schema.
"""

import pytest
from datetime import datetime, timezone, timedelta
from httpx import AsyncClient
from jose import jwt

from config.api import get_settings


def _sign(payload: dict) -> str:
    s = get_settings()
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def _expired_token(user_id: str = "test_user") -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    return _sign(payload)


def _tampered_token() -> str:
    """Take a valid token and flip one character in the signature."""
    s = get_settings()
    good = jwt.encode(
        {"user_id": "u1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        s.jwt_secret, algorithm=s.jwt_algorithm,
    )
    # Corrupt the last character of the signature
    return good[:-1] + ("X" if good[-1] != "X" else "Y")


def _assert_error_envelope(r, expected_status: int):
    """Verify response matches the standard ErrorEnvelope shape."""
    assert r.status_code == expected_status, r.text
    body = r.json()
    assert "error" in body, f"Missing 'error' key: {body}"
    err = body["error"]
    assert "code" in err
    assert "detail" in err
    assert "status" in err
    assert err["status"] == r.status_code


# Use any protected endpoint -- /api/v1/cases is convenient
PROTECTED = "/api/v1/cases"


@pytest.mark.asyncio
async def test_missing_auth_header_is_rejected(client: AsyncClient):
    r = await client.get(PROTECTED)
    assert r.status_code == 422  # FastAPI missing required header
    _assert_error_envelope(r, 422)


@pytest.mark.asyncio
async def test_malformed_bearer_prefix_is_rejected(client: AsyncClient):
    r = await client.get(PROTECTED, headers={"Authorization": "Token abc123"})
    assert r.status_code == 401
    _assert_error_envelope(r, 401)
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_tampered_token_is_rejected(client: AsyncClient):
    r = await client.get(PROTECTED, headers={"Authorization": f"Bearer {_tampered_token()}"})
    _assert_error_envelope(r, 401)


@pytest.mark.asyncio
async def test_expired_token_is_rejected(client: AsyncClient):
    r = await client.get(PROTECTED, headers={"Authorization": f"Bearer {_expired_token()}"})
    _assert_error_envelope(r, 401)


@pytest.mark.asyncio
async def test_token_without_user_id_is_rejected(client: AsyncClient):
    token = _sign({"sub": "no_user_id_field", "exp": datetime.now(timezone.utc) + timedelta(hours=1)})
    r = await client.get(PROTECTED, headers={"Authorization": f"Bearer {token}"})
    _assert_error_envelope(r, 401)
    assert "user_id" in r.json()["error"]["detail"]
