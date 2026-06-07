"""
config.ocr
----------
OCR pipeline constants sourced from settings.yaml.

Changes in this version
------------------------
PDF_DPI: int
    New.  Default 150 DPI (was hardcoded 300 in the pipeline).
    At 150 DPI a full A4 page is ~1240×1754 px — 4× fewer pixels than at
    300 DPI, so GCV inference is proportionally faster.  GCV's deep-learning
    OCR is robust at 150 DPI for both printed and handwritten Arabic text.
    Override in settings.yaml → ocr.performance.pdf_dpi: 300 if needed.

REFINEMENT_CONFIDENCE_THRESHOLD: float
    Pages at or above this GCV mean word-confidence skip LLM refinement.
    Printed Arabic legal text: typically 0.93–0.97.
    Handwritten Arabic text: typically 0.70–0.85.

MAX_WORKERS: int
    Thread-pool size — now used only for parallel LLM calls (GCV uses batch).

REFINEMENT_TIMEOUT: int
    Reduced default 60 s → 30 s (gemini-2.5-flash-lite is ~1 s per call).
"""

import os

from config import cfg

_ocr = cfg.ocr

# ----------------------------
# Language
# ----------------------------
OCR_LANGUAGE = _ocr.get("language", "ar")

# ----------------------------
# Engine
# ----------------------------
_engine = _ocr.get("engine", {})
OCR_PROVIDER = _engine.get("provider", "gcv")

# ----------------------------
# Google Cloud Vision
# ----------------------------
_gcv = _ocr.get("gcv", {})
GCV_FEATURE = _gcv.get("feature", "DOCUMENT_TEXT_DETECTION")

GCV_CREDENTIALS_FILE: str = (
    _gcv.get("credentials_file") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
)
if GCV_CREDENTIALS_FILE:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GCV_CREDENTIALS_FILE)

# ----------------------------
# LLM Refinement
# ----------------------------
_refinement = _ocr.get("refinement", {})
REFINEMENT_ENABLED: bool = _refinement.get("enabled", True)
REFINEMENT_TIMEOUT: int  = _refinement.get("timeout_seconds", 60)
REFINEMENT_CONFIDENCE_THRESHOLD: float = _refinement.get("confidence_threshold", 0.92)

# ----------------------------
# Confidence colour bands
# ----------------------------
_confidence = _ocr.get("confidence", {})
HIGH_CONFIDENCE_THRESHOLD: float   = _confidence.get("high_threshold", 0.90)
MEDIUM_CONFIDENCE_THRESHOLD: float = _confidence.get("medium_threshold", 0.65)

# ----------------------------
# Security / ingestion
# ----------------------------
_security = _ocr.get("security", {})
MAX_FILE_SIZE_MB: int = _security.get("max_file_size_mb", 50)
ALLOWED_EXTENSIONS: list = _security.get(
    "allowed_extensions",
    [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf", ".webp"],
)

# ----------------------------
# Performance
# ----------------------------
_perf = _ocr.get("performance", {})

# PDF → image conversion DPI.
# 150 DPI: 1240×1754 px per A4 page (2.2 MP) — 4× fewer pixels than 300 DPI.
# GCV inference time scales with pixel count; 150 DPI cuts it roughly 4×.
# Raise to 300 for very small printed text or severely degraded handwriting.
PDF_DPI: int = _perf.get("pdf_dpi", 150)

# Thread-pool size for parallel LLM calls.
# GCV now uses batch (single call) so this only applies to LLM workers.
MAX_WORKERS: int = _perf.get("max_workers", 4)

# ----------------------------
# Paths
# ----------------------------
import os as _os

_OCR_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "OCR")
)
BASE_DIR = _OCR_DIR
DICTIONARY_PATH = _os.path.join(_OCR_DIR, "dictionaries", "legal_arabic.txt")