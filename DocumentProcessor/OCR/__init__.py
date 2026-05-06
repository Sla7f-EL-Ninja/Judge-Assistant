"""
DocumentProcessor.OCR -- Arabic legal document OCR pipeline.
"""

from DocumentProcessor.OCR.models import OCRDocumentResult, OCRPageResult, WordConfidence
from DocumentProcessor.OCR.ocr_pipeline import run_ocr

__all__ = [
    "run_ocr",
    "OCRDocumentResult",
    "OCRPageResult",
    "WordConfidence",
]
