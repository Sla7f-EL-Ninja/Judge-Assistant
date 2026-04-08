"""
files.py

Schemas for file upload (POST /api/v1/files/upload).
"""

from datetime import datetime

from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    """Response after a successful file upload."""

    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="Detected MIME type")
    uploaded_at: datetime = Field(..., description="Upload timestamp (UTC)")
