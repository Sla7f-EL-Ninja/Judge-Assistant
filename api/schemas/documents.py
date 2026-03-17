"""
documents.py

Schemas for document ingestion (POST /api/v1/cases/{case_id}/documents).
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    file_ids: List[str] = Field(
        ..., min_length=1, description="File IDs to ingest into the case"
    )


class IngestResultItem(BaseModel):
    """Outcome of ingesting a single file."""

    file_id: str
    doc_type: str = ""
    classification: str = ""
    status: str = "success"


class IngestErrorItem(BaseModel):
    """Error detail for a file that failed ingestion."""

    file_id: str
    error: str


class IngestResponse(BaseModel):
    """Response from the document ingestion endpoint."""

    ingested: List[IngestResultItem] = Field(default_factory=list)
    errors: List[IngestErrorItem] = Field(default_factory=list)
