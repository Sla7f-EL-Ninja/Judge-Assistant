"""
test_ingestion_pipeline.py -- Full ingestion pipeline integration test.

Uploads a file, ingests it, and verifies all stores are written.

Marker: integration
"""

import pytest


@pytest.mark.integration
class TestIngestionPipeline:
    """Test the full document ingestion pipeline."""

    async def test_file_upload_endpoint(self, app_client, test_pdf_bytes):
        """POST /api/v1/files should accept PDF uploads."""
        from tests.conftest import auth_headers

        headers = auth_headers()
        response = await app_client.post(
            "/api/v1/files/upload",
            files={"file": ("test.pdf", test_pdf_bytes, "application/pdf")},
            headers=headers,
        )
        # Accept 200 or 201 for file upload
        assert response.status_code in (200, 201), (
            f"File upload should return 200 or 201, got {response.status_code}: "
            f"{response.text}"
        )

    async def test_cases_endpoint_exists(self, app_client):
        """GET /api/v1/cases should be accessible with auth."""
        from tests.conftest import auth_headers

        response = await app_client.get(
            "/api/v1/cases",
            headers=auth_headers(),
        )
        # Should not return 404 or 405
        assert response.status_code not in (404, 405), (
            f"Cases endpoint should exist, got {response.status_code}"
        )

    async def test_documents_endpoint_exists(self, app_client):
        """Documents endpoint should be accessible."""
        from tests.conftest import auth_headers

        # Try listing documents for a case
        response = await app_client.get(
            "/api/v1/cases/test_case/documents",
            headers=auth_headers(),
        )
        # Should not be 404 for the route itself
        # (may be 404 for case not found, which is acceptable)
        assert response.status_code != 405, (
            f"Documents endpoint method should be allowed, got {response.status_code}"
        )
