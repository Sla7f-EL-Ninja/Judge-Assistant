"""
DocumentProcessor.OCR.models
----------------------------
Pydantic data models for the OCR pipeline input/output contract.

Changes from previous version
------------------------------
- ``WordConfidence``: added ``band`` (UI colour band) and ``page_number``
  (needed when pages from multiple files are merged into one document).
- ``OCRPageResult``: added ``refined_text`` — the LLM-corrected version of
  ``normalized_text``.  This is the canonical text that gets classified,
  indexed, and shown to users.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class WordConfidence(BaseModel):
    """Per-word confidence score from the OCR engine."""

    word: str = Field(description="The recognised word text")
    confidence: float = Field(
        description="Word recognition confidence (0.0 – 1.0)",
        ge=0.0,
        le=1.0,
    )
    band: Literal["high", "mid", "low"] = Field(
        description=(
            "Confidence band used by the frontend for colour coding: "
            "'high' → green, 'mid' → orange, 'low' → red"
        ),
    )
    page_number: int = Field(
        default=1,
        description="1-based page index within the OCR run that produced this word",
    )


class OCRPageResult(BaseModel):
    """Result for a single page of OCR processing."""

    page_number: int = Field(description="1-based page index")
    raw_text: str = Field(
        default="",
        description="Raw text exactly as returned by the OCR engine",
    )
    normalized_text: str = Field(
        default="",
        description="Text after Arabic-Indic / Persian numeral normalisation",
    )
    refined_text: str = Field(
        default="",
        description=(
            "LLM-corrected Classical Arabic text. "
            "This is the canonical field used for classification, "
            "Qdrant indexing, and display.  Empty when refinement is disabled "
            "or the LLM call fails (normalized_text is used as fallback)."
        ),
    )
    perspective_corrected: bool = Field(
        default=False,
        description="Whether perspective correction was applied to this page",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Page-level mean confidence (0.0 – 1.0) from the OCR engine",
    )
    word_confidences: Optional[List[WordConfidence]] = Field(
        default=None,
        description="Per-word confidence scores for frontend colour highlighting",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if OCR processing failed for this page",
    )

    @property
    def canonical_text(self) -> str:
        """Return refined_text if available, otherwise normalized_text."""
        return self.refined_text if self.refined_text else self.normalized_text


class OCRDocumentResult(BaseModel):
    """Full document OCR result."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level metadata (filename, engine, timestamp, …)",
    )
    pages: List[OCRPageResult] = Field(
        default_factory=list,
        description="Per-page OCR results",
    )
