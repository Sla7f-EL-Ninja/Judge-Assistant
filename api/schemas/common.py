"""
common.py

Shared schemas: pagination, error responses, etc.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """Query parameters for paginated list endpoints."""

    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(20, ge=1, le=100, description="Max records to return")


class ErrorDetail(BaseModel):
    """Inner error object with machine-readable code and HTTP status."""

    code: str = Field(..., description="Machine-readable error code (e.g. CASE_NOT_FOUND)")
    detail: str = Field(..., description="Human-readable error message")
    status: int = Field(..., description="HTTP status code (mirrors the response status)")


class ErrorEnvelope(BaseModel):
    """Standard error response envelope used by all error handlers."""

    error: ErrorDetail


class ErrorResponse(BaseModel):
    """Deprecated -- use ErrorEnvelope instead.

    Kept for backwards compatibility during transition.
    """

    detail: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Machine-readable error code")


class MessageResponse(BaseModel):
    """Simple acknowledgement response."""

    message: str
