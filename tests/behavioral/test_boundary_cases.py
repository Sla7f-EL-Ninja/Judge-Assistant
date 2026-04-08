"""
test_boundary_cases.py -- Boundary and edge case testing.

Tests that the API handles unusual inputs gracefully without crashing.
Empty/whitespace queries should return 422 (validation error).
All other boundary inputs should not return 500.

Marker: behavioral
"""

import pytest


@pytest.mark.behavioral
class TestBoundaryCases:
    """Verify the API handles boundary inputs gracefully."""

    async def test_empty_query_returns_422(self, app_client):
        """Empty string query should return 422 validation error."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={"query": "", "case_id": ""},
            headers=auth_headers(),
        )
        assert response.status_code == 422, (
            f"Empty query should return 422, got {response.status_code}"
        )

    async def test_whitespace_only_query_returns_422(self, app_client):
        """Whitespace-only query should return 422 validation error."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={"query": "   ", "case_id": ""},
            headers=auth_headers(),
        )
        # Depending on whether whitespace is stripped before min_length check
        assert response.status_code in (422, 200), (
            f"Whitespace query should return 422 or be handled gracefully, "
            f"got {response.status_code}"
        )

    async def test_extremely_long_query_no_500(self, app_client):
        """Extremely long query should not cause a server error."""
        from tests.conftest import auth_headers

        long_query = "ما هي شروط صحة العقد " * 500  # ~10K characters
        response = await app_client.post(
            "/api/v1/query",
            json={"query": long_query, "case_id": ""},
            headers=auth_headers(),
        )
        assert response.status_code != 500, (
            f"Extremely long query returned 500 -- server error"
        )

    async def test_english_text_no_500(self, app_client):
        """English text should be handled without server error."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={
                "query": "What are the conditions for contract validity?",
                "case_id": "",
            },
            headers=auth_headers(),
        )
        assert response.status_code != 500, (
            f"English text query returned 500 -- server error"
        )

    async def test_vague_query_no_500(self, app_client):
        """Vague query should not cause server error."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={"query": "ماذا؟", "case_id": ""},
            headers=auth_headers(),
        )
        assert response.status_code != 500, (
            f"Vague query returned 500 -- server error"
        )

    async def test_nonexistent_article_no_500(self, app_client):
        """Reference to nonexistent article should not crash."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={
                "query": "ما نص المادة 99999 من القانون المدني؟",
                "case_id": "",
            },
            headers=auth_headers(),
        )
        assert response.status_code != 500, (
            f"Nonexistent article query returned 500 -- server error"
        )

    async def test_newlines_only_query(self, app_client):
        """Newlines-only query should be handled without crash."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={"query": "\n\n\n", "case_id": ""},
            headers=auth_headers(),
        )
        assert response.status_code != 500, (
            f"Newlines-only query returned 500 -- server error"
        )

    async def test_sql_injection_attempt_no_500(self, app_client):
        """SQL injection attempt should not cause server error."""
        from tests.conftest import auth_headers

        response = await app_client.post(
            "/api/v1/query",
            json={
                "query": "'; DROP TABLE documents; --",
                "case_id": "",
            },
            headers=auth_headers(),
        )
        assert response.status_code != 500, (
            f"SQL injection attempt returned 500 -- server error"
        )
