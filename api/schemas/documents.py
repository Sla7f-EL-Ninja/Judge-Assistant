"""
documents.py

Schemas for document endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    file_ids: List[str] = Field(
        ..., min_length=1, description="File IDs to ingest into the case"
    )


class IngestResultItem(BaseModel):
    """Outcome of ingesting a single file."""

    file_id: str
    doc_type: Optional[str] = None
    classification: Dict[str, Any] = Field(default_factory=dict)
    status: str


class IngestErrorItem(BaseModel):
    """Error detail for a file that failed ingestion."""

    file_id: str
    error: str
    status: str = "failed"


class IngestResponse(BaseModel):
    """Response from the document ingestion endpoint."""

    ingested: List[IngestResultItem] = Field(default_factory=list)
    errors: List[IngestErrorItem] = Field(default_factory=list)


class DocumentItem(BaseModel):
    id: str
    title: str
    source_file: str
    doc_type: Optional[str] = None
    file_type: Optional[str] = None
    file_id: Optional[str] = None
    created_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total: int


class ClassificationDetail(BaseModel):
    final_type: str = ""
    confidence: float = 0.0
    explanation: str = ""


class DocumentDetailResponse(BaseModel):
    id: str
    title: str
    source_file: str
    doc_type: Optional[str] = None
    file_type: Optional[str] = None
    file_id: Optional[str] = None
    created_at: Optional[datetime] = None
    text_excerpt: str = ""
    classification: Optional[ClassificationDetail] = None
    storage_backend: Optional[str] = None
    minio_object: Optional[str] = None
    qdrant_chunks: int = 0
    corrected: bool = False
    corrected_at: Optional[datetime] = None


class OCRTextResponse(BaseModel):
    doc_id: str
    file_id: Optional[str] = None
    file_type: Optional[str] = None
    source_file: str = ""
    text: str = ""
    classification: Optional[ClassificationDetail] = None
    corrected: bool = False
    corrected_at: Optional[datetime] = None
    original_text: Optional[str] = None


class OCRCorrectionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    corrected_by: Optional[str] = None
