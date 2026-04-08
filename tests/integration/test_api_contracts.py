"""
test_api_contracts.py -- Validate API endpoint response contracts.

Uses httpx AsyncClient with ASGITransport to test the FastAPI application
without a running server.

Marker: integration
"""

import json

import pytest


@pytest.mark.integration
class TestHealthEndpoint:
    """Validate GET /api/v1/health contract."""

    async def test_health_returns_200(self, app_client):
        """Health endpoint should return 200."""
        response = await app_client.get("/api/v1/health")
        assert response.status_code == 200, (
            f"Health endpoint should return 200, got {response.status_code}"
        )

    async def test_health_response_shape(self, app_client):
        """Health response must contain status, version, and dependencies."""
        response = await app_client.get("/api/v1/health")
        data = response.json()
        assert "status" in data, "Health response must include 'status'"
        assert data["status"] in ("healthy", "degraded"), (
            f"Health status must be 'healthy' or 'degraded', got '{data['status']}'"
        )
        assert "version" in data, "Health response must include 'version'"
        assert isinstance(data["version"], str), (
            "Health version must be a string"
        )
        assert "dependencies" in data, "Health response must include 'dependencies'"
        assert isinstance(data["dependencies"], dict), (
            "Health dependencies must be a dict"
        )


@pytest.mark.integration
class TestQueryEndpoint:
    """Validate POST /api/v1/query SSE contract."""

    async def test_query_requires_auth(self, app_client):
        """Query endpoint should reject requests without auth."""
        response = await app_client.post(
            "/api/v1/query",
            json={"query": "test", "case_id": ""},
        )
        assert response.status_code == 401, (
            f"Query endpoint should return 401 without auth, got {response.status_code}"
        )

    async def test_query_validates_empty_query(self, app_client):
        """Query endpoint should reject empty query with 422."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={"query": "", "case_id": ""},
            headers=auth_headers(),
        )
        assert response.status_code == 422, (
            f"Empty query should return 422, got {response.status_code}"
        )

    async def test_query_returns_sse_content_type(self, app_client):
        """Query endpoint should return text/event-stream content type."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={"query": "ما هي شروط صحة العقد؟", "case_id": ""},
            headers=auth_headers(),
        )
        assert response.status_code == 200, (
            f"Valid query should return 200, got {response.status_code}"
        )
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"Query response must be text/event-stream, got '{content_type}'"
        )

    async def test_contract_never_returns_500_on_valid_input(self, app_client):
        """10 valid Arabic queries should never produce a 500 response."""
        from tests.conftest import auth_headers

        headers = auth_headers()
        queries = [
            "ما هي شروط صحة العقد؟",
            "ما أحكام المسؤولية التقصيرية؟",
            "ما هي أحكام الفسخ؟",
            "ما أحكام حق الحبس؟",
            "ما هي شروط الوفاء بالالتزام؟",
            "ما أحكام التقادم المسقط؟",
            "ما هي أحكام الكفالة؟",
            "ما أحكام الإثراء بلا سبب؟",
            "ما هي أحكام الفضالة؟",
            "ما أحكام المقاصة؟",
        ]
        for query in queries:
            response = await app_client.post(
                "/api/v1/query",
                json={"query": query, "case_id": ""},
                headers=headers,
            )
            assert response.status_code != 500, (
                f"Valid query '{query}' returned 500 -- server error"
            )
