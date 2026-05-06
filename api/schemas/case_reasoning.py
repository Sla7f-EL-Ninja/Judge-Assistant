"""
case_reasoning.py

Schemas for the case reasoning endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CaseReasoningResponse(BaseModel):
    case_id: str
    final_report: str = ""
    issue_analyses: List[Dict[str, Any]] = Field(default_factory=list)
    cross_issue_relationships: List[Dict[str, Any]] = Field(default_factory=list)
    consistency_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    reconciliation_paragraphs: List[str] = Field(default_factory=list)
    per_issue_confidence: List[Dict[str, Any]] = Field(default_factory=list)
    case_level_confidence: Dict[str, Any] = Field(default_factory=dict)
    generated_at: Optional[datetime] = None

class GenerateCaseReasoningResponse(BaseModel):
    """Lightweight acknowledgement returned by POST /case-reasoning/generate."""

    case_id: str
    issues_analyzed: int
    message: str