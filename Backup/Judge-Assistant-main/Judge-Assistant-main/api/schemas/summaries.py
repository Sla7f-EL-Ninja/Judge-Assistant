"""
summaries.py

Schemas for the summary read endpoint.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class SummaryResponse(BaseModel):
    """Stored case summary."""

    case_id: str
    summary: str
    generated_at: datetime
    sources: List[str] = Field(default_factory=list)
