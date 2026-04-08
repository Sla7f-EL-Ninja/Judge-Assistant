"""
query.py

Schemas for the supervisor query endpoint (POST /api/v1/query).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming judge query."""

    query: str = Field(..., min_length=1, description="The judge's question")
    case_id: str = Field("", description="Active case identifier")
    conversation_id: Optional[str] = Field(
        None, description="Existing conversation to continue"
    )


class QueryProgressEvent(BaseModel):
    """SSE progress event emitted while the supervisor graph runs."""

    step: str = Field(..., description="Graph node name")
    status: str = Field(..., description="running | done")
    detail: Optional[Dict[str, Any]] = Field(
        None, description="Extra info (intent, agents, validation result, etc.)"
    )


class QueryResult(BaseModel):
    """Final result payload sent via SSE."""

    final_response: str
    sources: List[str] = []
    intent: str = ""
    agents_used: List[str] = []
    conversation_id: str = ""
