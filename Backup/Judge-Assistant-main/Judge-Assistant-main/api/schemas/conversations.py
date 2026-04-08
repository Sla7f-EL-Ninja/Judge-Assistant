"""
conversations.py

Schemas for conversation history endpoints.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class TurnSchema(BaseModel):
    """A single conversation turn."""

    turn_number: int
    query: str
    response: str
    intent: str = ""
    agents_used: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    timestamp: datetime


class ConversationResponse(BaseModel):
    """Full conversation object."""

    id: str = Field(..., alias="_id")
    case_id: str
    user_id: str
    turns: List[TurnSchema] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"populate_by_name": True}


class ConversationListResponse(BaseModel):
    """Paginated list of conversations for a case."""

    conversations: List[ConversationResponse]
    total: int
