"""
test_auth.py

Tests for JWT authentication dependency.
"""

from jose import jwt
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from api.dependencies import get_db, get_settings as dep_get_settings
from api.tests.conftest import FakeDatabase, TEST_JWT_SECRET


def _make_client_no_auth_override():
    """Create a TestClient WITHOUT overriding get_current_user,
    so we can test actual JWT validation."""
    app = create_app()
    settings = Settings(
        jwt_secret=TEST_JWT_SECRET,
        jwt_algorithm="HS256",
        mongo_uri="mongodb://localhost:27017/",
        mongo_db="test_db",
        upload_dir="/tmp/test_uploads",
        debug=True,
    )
    fake_db = FakeDatabase()

    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[dep_get_settings] = lambda: settings

    return TestClient(app, raise_server_exceptions=False)


def test_missing_auth_header():
    """Requests without Authorization header should get 422."""
    client = _make_client_no_auth_override()
    resp = client.get("/api/v1/cases")
    assert resp.status_code == 422


def test_invalid_bearer_prefix():
    """Authorization header without 'Bearer ' prefix should get 401."""
    client = _make_client_no_auth_override()
    resp = client.get(
        "/api/v1/cases",
        headers={"Authorization": "Token some-token"},
    )
    assert resp.status_code == 401


def test_invalid_token():
    """An invalid JWT should return 401."""
    client = _make_client_no_auth_override()
    resp = client.get(
        "/api/v1/cases",
        headers={"Authorization": "Bearer invalid.jwt.token"},
    )
    assert resp.status_code == 401


def test_token_missing_user_id():
    """A JWT without user_id in payload should return 401."""
    client = _make_client_no_auth_override()
    token = jwt.encode({"role": "admin"}, TEST_JWT_SECRET, algorithm="HS256")
    resp = client.get(
        "/api/v1/cases",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_valid_token():
    """A valid JWT with user_id should succeed."""
    client = _make_client_no_auth_override()
    token = jwt.encode(
        {"user_id": "test_user_42"}, TEST_JWT_SECRET, algorithm="HS256"
    )
    resp = client.get(
        "/api/v1/cases",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
