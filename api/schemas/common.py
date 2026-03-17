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


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Machine-readable error code")


class MessageResponse(BaseModel):
    """Simple acknowledgement response."""

    message: str
