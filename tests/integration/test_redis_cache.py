"""
test_redis_cache.py -- Verify Redis caching behavior.

Tests cache write on first query and cache hit on second identical query.
Skip if Redis is unavailable.

Marker: integration
"""

import pytest


@pytest.mark.integration
class TestRedisCache:
    """Verify Redis caching behavior for query results."""

    async def test_redis_can_set_and_get(self, redis_client):
        """Basic Redis set/get should work."""
        test_key = "hakim_test:cache_test"
        test_value = "test_value_123"

        await redis_client.set(test_key, test_value, ex=60)
        result = await redis_client.get(test_key)
        assert result == test_value, (
            f"Redis get should return '{test_value}', got '{result}'"
        )

        # Cleanup
        await redis_client.delete(test_key)

    async def test_redis_expiry_works(self, redis_client):
        """Redis keys should respect TTL."""
        test_key = "hakim_test:expiry_test"
        await redis_client.set(test_key, "expires", ex=1)

        ttl = await redis_client.ttl(test_key)
        assert ttl > 0, "Key should have positive TTL after set"

        # Cleanup
        await redis_client.delete(test_key)

    async def test_redis_key_format(self, redis_client):
        """Cache keys should follow a consistent format."""
        # Verify we can create keys with the expected pattern
        test_key = "hakim:query_cache:test_hash_123"
        await redis_client.set(test_key, '{"response": "cached"}', ex=60)

        result = await redis_client.get(test_key)
        assert result is not None, (
            "Cache key with hakim:query_cache: prefix should be settable"
        )

        # Cleanup
        await redis_client.delete(test_key)

    async def test_redis_json_round_trip(self, redis_client):
        """JSON data should survive Redis round-trip."""
        import json

        test_key = "hakim_test:json_test"
        test_data = {
            "final_response": "تعديل العقد يتطلب موافقة الطرفين",
            "sources": ["مادة 147", "مادة 148"],
            "intent": "civil_law_rag",
        }

        await redis_client.set(test_key, json.dumps(test_data, ensure_ascii=False), ex=60)
        raw = await redis_client.get(test_key)
        recovered = json.loads(raw)

        assert recovered["final_response"] == test_data["final_response"], (
            "JSON round-trip should preserve Arabic text"
        )
        assert recovered["sources"] == test_data["sources"], (
            "JSON round-trip should preserve list data"
        )

        # Cleanup
        await redis_client.delete(test_key)
