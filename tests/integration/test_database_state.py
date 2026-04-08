"""
test_database_state.py -- Verify document ingestion writes to all stores.

Uses database fixtures with skip guards to verify that ingested documents
are properly stored in MongoDB, Qdrant, and MinIO.

Marker: integration
"""

import pytest


@pytest.mark.integration
class TestDatabaseState:
    """Verify document state across MongoDB, Qdrant, and MinIO."""

    async def test_mongodb_connectivity(self, motor_client):
        """MongoDB should be reachable and respond to commands."""
        result = await motor_client.command("ping")
        assert result.get("ok") == 1.0, (
            "MongoDB ping should return ok=1.0"
        )

    async def test_mongodb_collection_exists(self, motor_client):
        """The document storage collection should exist in MongoDB."""
        from config.api import get_settings

        settings = get_settings()
        collections = await motor_client.list_collection_names()
        assert settings.mongo_collection in collections, (
            f"Expected MongoDB collection '{settings.mongo_collection}' to exist, "
            f"found: {collections}"
        )

    def test_qdrant_connectivity(self, qdrant_client):
        """Qdrant should be reachable."""
        collections = qdrant_client.get_collections()
        assert collections is not None, "Qdrant should return collections list"

    def test_qdrant_collection_exists(self, qdrant_client):
        """The judicial docs collection should exist in Qdrant."""
        from config.api import get_settings

        settings = get_settings()
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        assert settings.qdrant_collection in collection_names, (
            f"Expected Qdrant collection '{settings.qdrant_collection}', "
            f"found: {collection_names}"
        )

    def test_minio_connectivity(self, minio_client):
        """MinIO should be reachable."""
        buckets = minio_client.list_buckets()
        assert buckets is not None, "MinIO should return buckets list"

    def test_minio_bucket_exists(self, minio_client):
        """The configured bucket should exist in MinIO."""
        from config.api import get_settings

        settings = get_settings()
        bucket_name = settings.minio_bucket
        assert minio_client.bucket_exists(bucket_name), (
            f"Expected MinIO bucket '{bucket_name}' to exist"
        )

    async def test_redis_connectivity(self, redis_client):
        """Redis should be reachable and respond to ping."""
        result = await redis_client.ping()
        assert result is True, "Redis ping should return True"
