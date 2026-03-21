"""
test_health.py

Tests for GET /api/v1/health
"""


def test_health_returns_200(client):
    """Health endpoint should return 200 with status and version."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
    assert "dependencies" in data


def test_health_has_mongodb_dependency(client):
    """Health response should include mongodb dependency status."""
    resp = client.get("/api/v1/health")
    data = resp.json()
    assert "mongodb" in data["dependencies"]
