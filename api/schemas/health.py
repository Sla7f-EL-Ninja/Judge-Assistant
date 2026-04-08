"""
health.py

Schema for the GET /api/v1/health endpoint response.
"""

from typing import Dict

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(..., description="Overall health status: healthy | degraded")
    version: str = Field(..., description="Application version string")
    dependencies: Dict[str, str] = Field(
        ..., description="Per-dependency connectivity status"
    )
