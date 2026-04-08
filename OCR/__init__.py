"""
OCR -- Arabic legal document OCR pipeline.

Uses the QARI model (Qwen2VL-based VLM) for full-page OCR with
confidence scoring.
"""

from OCR.models import OCRDocumentResult, OCRPageResult, WordConfidence
from OCR.ocr_pipeline import process_document

__all__ = [
    "process_document",
    "OCRDocumentResult",
    "OCRPageResult",
    "WordConfidence",
]
