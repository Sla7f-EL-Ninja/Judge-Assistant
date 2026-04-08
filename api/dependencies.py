"""
dependencies.py

FastAPI dependencies used across routers.

- ``get_current_user`` -- validates the JWT and returns the user_id.
- ``get_db`` -- returns the async MongoDB database handle.
- ``get_settings`` -- returns the application settings singleton.
- ``get_qdrant`` -- returns the Qdrant vector store client.
- ``get_redis_client`` -- returns the async Redis client.
- ``get_pg_session`` -- yields a PostgreSQL async session.
"""

from typing import AsyncGenerator, Optional

from fastapi import Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from config.api import Settings, get_settings as _get_settings
from api.db.mongodb import get_database
from api.services.auth import AuthError, decode_token


def get_settings() -> Settings:
    """Return the application settings."""
    return _get_settings()


def get_db() -> AsyncIOMotorDatabase:
    """Return the active MongoDB database handle."""
    return get_database()


async def get_current_user(
    authorization: str = Header(..., description="Bearer <JWT>"),
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate the JWT from the Authorization header and return the user_id.

    Raises HTTP 401 on invalid or missing tokens.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '",
        )

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token, settings)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.detail,
        ) from exc

    return payload["user_id"]


def get_qdrant():
    """Return the active Qdrant client."""
    from api.db.qdrant import get_qdrant_client

    return get_qdrant_client()


def get_redis_client():
    """Return the active Redis client (may be None if Redis is down)."""
    from api.db.redis import get_redis

    return get_redis()


async def get_pg_session() -> AsyncGenerator:
    """Yield a PostgreSQL async session."""
    from api.db.postgres import get_async_session

    async for session in get_async_session():
        yield session
