"""
test_cache_speedup.py -- Verify Redis cache provides meaningful speedup.

Runs the same query twice, measuring wall-clock time. The warm (cached)
run should be significantly faster than the cold run.

Marker: performance
"""

import time

import pytest


@pytest.mark.performance
class TestCacheSpeedup:
    """Verify that caching provides meaningful query speedup."""

    async def test_warm_run_faster_than_cold(self, app_client, redis_client):
        """Cached query should be at least 2x faster than uncached."""
        from tests.conftest import auth_headers

        headers = auth_headers()
        query = "ما هي شروط صحة العقد في القانون المدني المصري؟"
        payload = {"query": query, "case_id": ""}

        # Cold run
        start_cold = time.monotonic()
        response_cold = await app_client.post(
            "/api/v1/query",
            json=payload,
            headers=headers,
        )
        cold_time = time.monotonic() - start_cold

        assert response_cold.status_code == 200, (
            f"Cold query failed with status {response_cold.status_code}"
        )

        # Warm run (should hit cache)
        start_warm = time.monotonic()
        response_warm = await app_client.post(
            "/api/v1/query",
            json=payload,
            headers=headers,
        )
        warm_time = time.monotonic() - start_warm

        assert response_warm.status_code == 200, (
            f"Warm query failed with status {response_warm.status_code}"
        )

        # Assert speedup: warm <= cold * 0.5
        # If cold run is too fast (< 1s), skip the comparison
        if cold_time < 1.0:
            pytest.skip(
                f"Cold run too fast ({cold_time:.2f}s) to measure meaningful speedup"
            )

        assert warm_time <= cold_time * 0.5, (
            f"Cache speedup insufficient: cold={cold_time:.2f}s, warm={warm_time:.2f}s "
            f"(expected warm <= {cold_time * 0.5:.2f}s)"
        )
