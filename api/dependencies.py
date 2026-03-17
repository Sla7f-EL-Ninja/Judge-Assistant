"""
dependencies.py

FastAPI dependencies used across routers.

- ``get_current_user`` -- validates the JWT and returns the user_id.
- ``get_db`` -- returns the async MongoDB database handle.
- ``get_settings`` -- returns the application settings singleton.
"""

from fastapi import Depends, Header, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api.config import Settings, get_settings as _get_settings
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
