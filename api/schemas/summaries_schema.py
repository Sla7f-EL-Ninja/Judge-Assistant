"""
summaries.py

Schemas for the summary endpoints.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SummaryResponse(BaseModel):
    """Stored case summary returned by GET /summary."""

    case_id: str
    summary: str
    generated_at: datetime
    sources: List[str] = Field(default_factory=list)


class GenerateSummaryResponse(BaseModel):
    """Lightweight acknowledgement returned by POST /summary/generate."""

    case_id: str
    sources_count: int
    message: str
