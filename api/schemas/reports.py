"""
reports.py

Schemas for async report generation endpoints.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from api.schemas.case_reasoning import CaseReasoningResponse
from api.schemas.summaries import SummaryResponse


class ReportJobAck(BaseModel):
    job_id: str
    status: str = "queued"
    case_id: str


class ReportJobSummary(BaseModel):
    report_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class ReportJobStatusResponse(BaseModel):
    report_id: str
    case_id: str
    status: str  # queued | running | completed | failed
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    summary: Optional[SummaryResponse] = None
    case_reasoning: Optional[CaseReasoningResponse] = None


class ReportJobListResponse(BaseModel):
    jobs: List[ReportJobSummary]
    total: int
