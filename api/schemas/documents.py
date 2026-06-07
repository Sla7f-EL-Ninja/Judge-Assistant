"""
api/schemas/documents.py

Schemas for document endpoints.

Changes
-------
- Added ``WordConfidenceItem`` — per-word OCR confidence for frontend highlighting.
- ``OCRTextResponse`` now includes ``word_confidences: List[WordConfidenceItem]``.
  Each item carries a ``band`` field ("high" | "mid" | "low") so the React
  frontend can apply colour without doing any threshold arithmetic.
- ``raw_ocr_text`` is intentionally NOT exposed here — it is an internal
  audit field stored in MongoDB only.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestGroup(BaseModel):
    """A set of file IDs that form one document (ordered, page order)."""

    file_ids: List[str] = Field(..., min_length=1)


class IngestRequest(BaseModel):
    """Request body for document ingestion.

    Use ``file_ids`` for backwards-compatible single-file-per-document ingestion.
    Use ``groups`` when multiple files form a single multi-page document.
    Exactly one of the two must be provided.
    """

    file_ids: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        description="Legacy: each file becomes its own document",
    )
    groups: Optional[List[IngestGroup]] = Field(
        default=None,
        min_length=1,
        description="Each group (ordered file IDs) becomes one document",
    )

    @property
    def resolved_groups(self) -> List[IngestGroup]:
        """Normalise both request forms to a list of IngestGroup."""
        if self.groups is not None:
            return self.groups
        if self.file_ids is not None:
            return [IngestGroup(file_ids=[fid]) for fid in self.file_ids]
        raise ValueError("Provide either file_ids or groups")

    def model_post_init(self, __context) -> None:
        if self.file_ids is None and self.groups is None:
            raise ValueError("Provide either file_ids or groups")
        if self.file_ids is not None and self.groups is not None:
            raise ValueError("Provide file_ids or groups, not both")


class IngestResultItem(BaseModel):
    """Outcome of ingesting a group of files into one document."""

    file_ids: List[str]
    file_id: Optional[str] = None          # deprecated alias — file_ids[0]
    doc_id: Optional[str] = None
    doc_type: Optional[str] = None
    classification: Dict[str, Any] = Field(default_factory=dict)
    status: str


class IngestErrorItem(BaseModel):
    """Error detail for a group that failed ingestion."""

    file_ids: List[str]
    file_id: Optional[str] = None          # deprecated alias
    error: str
    status: str = "failed"


class IngestResponse(BaseModel):
    """Response from the document ingestion endpoint."""

    ingested: List[IngestResultItem] = Field(default_factory=list)
    errors: List[IngestErrorItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Document list / detail
# ---------------------------------------------------------------------------

class DocumentItem(BaseModel):
    id: str
    title: str
    source_file: str
    source_files: List[str] = Field(default_factory=list)
    doc_type: Optional[str] = None
    file_type: Optional[str] = None
    file_ids: List[str] = Field(default_factory=list)
    file_id: Optional[str] = None          # deprecated alias — file_ids[0]
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
    source_files: List[str] = Field(default_factory=list)
    doc_type: Optional[str] = None
    file_type: Optional[str] = None
    file_ids: List[str] = Field(default_factory=list)
    file_id: Optional[str] = None          # deprecated alias — file_ids[0]
    created_at: Optional[datetime] = None
    text_excerpt: str = ""
    classification: Optional[ClassificationDetail] = None
    storage_backend: Optional[str] = None
    minio_object: Optional[str] = None
    qdrant_chunks: int = 0
    corrected: bool = False
    corrected_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# OCR — word confidence
# ---------------------------------------------------------------------------

class WordConfidenceItem(BaseModel):
    """Per-word OCR confidence item, designed for frontend colour highlighting.

    ``band`` is pre-computed by the OCR engine so the React component never
    needs to know the raw threshold values::

        band == "high"  →  green   (confidence ≥ 0.90)
        band == "mid"   →  orange  (0.65 ≤ confidence < 0.90)
        band == "low"   →  red     (confidence < 0.65)
    """

    word: str
    confidence: float = Field(ge=0.0, le=1.0)
    band: Literal["high", "mid", "low"]
    page_number: int = Field(default=1, description="1-based page index")


# ---------------------------------------------------------------------------
# OCR text responses
# ---------------------------------------------------------------------------

class OCRTextResponse(BaseModel):
    doc_id: str
    file_ids: List[str] = Field(default_factory=list)
    file_id: Optional[str] = None          # deprecated alias — file_ids[0]
    file_type: Optional[str] = None
    source_file: str = ""
    source_files: List[str] = Field(default_factory=list)
    text: str = ""
    word_confidences: List[WordConfidenceItem] = Field(
        default_factory=list,
        description=(
            "Per-word OCR confidence list.  Use the ``band`` field to apply "
            "colour coding in the UI.  Empty for non-OCR documents (plain text / "
            "machine-readable PDFs) and when no confidence data was captured."
        ),
    )
    classification: Optional[ClassificationDetail] = None
    corrected: bool = False
    corrected_at: Optional[datetime] = None
    original_text: Optional[str] = None


class OCRCorrectionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    corrected_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Bulk OCR correction
# ---------------------------------------------------------------------------

class BulkOCRCorrectionItem(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    corrected_by: Optional[str] = None


class BulkOCRCorrectionRequest(BaseModel):
    corrections: List[BulkOCRCorrectionItem] = Field(..., min_length=1)
    corrected_by: Optional[str] = None

    @model_validator(mode="after")
    def _validate_corrections(self) -> "BulkOCRCorrectionRequest":
        from config.api import get_settings

        cap = get_settings().bulk_ocr_max_items
        if len(self.corrections) > cap:
            raise ValueError(f"corrections list exceeds maximum of {cap} items")
        ids = [item.doc_id for item in self.corrections]
        if len(ids) != len(set(ids)):
            raise ValueError("corrections contains duplicate doc_id values")
        return self


class BulkOCRCorrectionResultItem(BaseModel):
    doc_id: str
    status: Literal["success", "failed"]
    result: Optional[OCRTextResponse] = None
    error: Optional[Dict[str, str]] = None


class BulkOCRCorrectionResponse(BaseModel):
    results: List[BulkOCRCorrectionResultItem]
    succeeded: int
    failed: int
