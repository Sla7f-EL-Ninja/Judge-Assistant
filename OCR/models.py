"""
OCR.models
----------
Pydantic data models for the OCR pipeline input/output contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WordConfidence(BaseModel):
    """Per-word confidence score (stretch goal)."""

    word: str = Field(description="The decoded word text")
    confidence: float = Field(
        description="Average token probability for this word (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )


class OCRPageResult(BaseModel):
    """Result for a single page of OCR processing."""

    page_number: int = Field(description="1-based page index")
    raw_text: str = Field(default="", description="Raw text from the OCR engine")
    normalized_text: str = Field(
        default="",
        description="Text after numeral normalization and post-processing",
    )
    perspective_corrected: bool = Field(
        default=False,
        description="Whether perspective correction was applied to this page",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Page-level confidence score (0.0 to 1.0)",
    )
    word_confidences: Optional[List[WordConfidence]] = Field(
        default=None,
        description="Per-word confidence scores (stretch goal, may be None)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed for this page",
    )


class OCRDocumentResult(BaseModel):
    """Full document OCR result."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level metadata (filename, model, timestamp, etc.)",
    )
    pages: List[OCRPageResult] = Field(
        default_factory=list,
        description="Per-page OCR results",
    )
