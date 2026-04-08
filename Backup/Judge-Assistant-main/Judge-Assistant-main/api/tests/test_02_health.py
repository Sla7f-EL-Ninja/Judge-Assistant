"""
test_health.py

Verify that the server boots, MongoDB is reachable, and Qdrant is connected.
This is always the first thing to confirm before running anything else.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_health_response_schema(client: AsyncClient):
    """Verify that the health response matches HealthResponse schema."""
    r = await client.get("/api/v1/health")
    body = r.json()
    assert "status" in body, f"Missing 'status': {body}"
    assert "version" in body, f"Missing 'version': {body}"
    assert "dependencies" in body, f"Missing 'dependencies': {body}"
    assert isinstance(body["dependencies"], dict)


@pytest.mark.asyncio
async def test_health_mongodb_connected(client: AsyncClient):
    r = await client.get("/api/v1/health")
    body = r.json()
    assert body["dependencies"]["mongodb"] == "connected", (
        f"MongoDB not connected. Full response: {body}"
    )


@pytest.mark.asyncio
async def test_health_qdrant_connected(client: AsyncClient):
    r = await client.get("/api/v1/health")
    body = r.json()
    qdrant = body["dependencies"].get("qdrant")
    assert qdrant == "connected", (
        f"Qdrant status: '{qdrant}'. "
        f"Make sure Qdrant server is running and accessible."
    )


@pytest.mark.asyncio
async def test_health_overall_status(client: AsyncClient):
    r = await client.get("/api/v1/health")
    body = r.json()
    assert body["status"] == "healthy", (
        f"Overall status degraded. Dependencies: {body['dependencies']}"
    )
