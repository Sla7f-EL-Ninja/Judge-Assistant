"""
redis.py

Redis client management for caching, rate limiting, and session storage.

``connect_redis`` / ``close_redis`` are called from the FastAPI lifespan.
``get_redis`` returns the active async Redis client for dependency injection.

Features:
- LLM response caching (avoids redundant API calls)
- Rate limiting per user
- Active conversation context caching
"""

import logging
from typing import Optional

import redis.asyncio as aioredis

from config.api import Settings

logger = logging.getLogger(__name__)

_pool: Optional[aioredis.Redis] = None


async def connect_redis(settings: Settings) -> None:
    """Create the async Redis connection pool."""
    global _pool
    _pool = aioredis.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )

    # Verify connectivity
    try:
        await _pool.ping()
        logger.info("Redis connected at %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Redis connection failed: %s", exc)
        _pool = None
        raise


async def close_redis() -> None:
    """Close the Redis connection pool."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


def get_redis() -> Optional[aioredis.Redis]:
    """Return the active Redis client, or None if not connected."""
    return _pool


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

async def cache_get(key: str) -> Optional[str]:
    """Retrieve a cached value by key. Returns None on miss or if Redis is down."""
    if _pool is None:
        return None
    try:
        return await _pool.get(key)
    except Exception:
        return None


async def cache_set(key: str, value: str, ttl_seconds: int = 3600) -> None:
    """Store a value in cache with a TTL. Silently fails if Redis is down."""
    if _pool is None:
        return
    try:
        await _pool.setex(key, ttl_seconds, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rate limiting helpers
# ---------------------------------------------------------------------------

async def check_rate_limit(
    user_id: str,
    max_requests: int = 100,
    window_seconds: int = 60,
) -> bool:
    """Check if a user has exceeded their rate limit.

    Returns True if the request is allowed, False if rate-limited.
    Uses a sliding window counter pattern.
    """
    if _pool is None:
        return True  # If Redis is down, don't block requests

    key = f"rate_limit:{user_id}"
    try:
        current = await _pool.incr(key)
        if current == 1:
            await _pool.expire(key, window_seconds)
        return current <= max_requests
    except Exception:
        return True  # Fail open if Redis has issues
