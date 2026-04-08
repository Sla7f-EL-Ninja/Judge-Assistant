"""
auth.py

JWT token decoding and validation.

Tokens are issued by the Express backend and validated here using a
shared secret.
"""

from jose import JWTError, jwt

from typing import Optional

from config.api import Settings


class AuthError(Exception):
    """Raised when JWT validation fails."""

    def __init__(self, detail: str = "Could not validate credentials"):
        self.detail = detail


def decode_token(token: str, settings: Settings) -> dict:
    """Decode and validate a JWT, returning the payload.

    Raises ``AuthError`` on any failure.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: Optional[str] = payload.get("user_id")
        if user_id is None:
            raise AuthError("Token payload missing user_id")
        return payload
    except JWTError as exc:
        raise AuthError(f"Invalid token: {exc}") from exc
