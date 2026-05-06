"""
cases.py

Schemas for case management CRUD endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class CaseDocumentRef(BaseModel):
    """Reference to an ingested document within a case."""

    file_id: str
    filename: str
    classification: Dict[str, Any] = Field(default_factory=dict)
    ingested_at: Optional[datetime] = None
    # Add this validator block:
    @field_validator("classification", mode="before")
    @classmethod
    def handle_legacy_strings(cls, v):
        if isinstance(v, str):
            # If the DB returns a string (like ""), convert it to an empty dict
            return {} 
        return v


class CaseCreate(BaseModel):
    """Request body for creating a new case."""

    title: str = Field(..., min_length=1, description="Case title")
    description: str = Field("", description="Case description")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary case metadata"
    )


class CaseUpdate(BaseModel):
    """Request body for updating an existing case."""

    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = Field(
        None, pattern="^(active|archived|closed)$",
        description="Case status"
    )
    metadata: Optional[Dict[str, Any]] = None


class CaseResponse(BaseModel):
    """Full case object returned from the API."""

    id: str = Field(..., alias="_id")
    user_id: str
    title: str
    description: str = ""
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    documents: List[CaseDocumentRef] = Field(default_factory=list)
    conversation_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"populate_by_name": True}


class CaseListResponse(BaseModel):
    """Paginated list of cases."""

    cases: List[CaseResponse]
    total: int
