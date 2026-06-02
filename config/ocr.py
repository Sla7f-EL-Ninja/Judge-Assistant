# """
# config.ocr
# ----------
# OCR pipeline constants sourced from ``settings.yaml`` via the centralised
# ``config`` module.

# Removed from previous version
# ------------------------------
# - PREPROCESSING_ENABLED  (preprocessing stage removed — GCV handles it)
# - ENABLE_RESOLUTION_CHECK, MIN_DPI, ENABLE_DESKEW, ENABLE_DENOISE,
#   ENABLE_BORDER_REMOVAL, ENABLE_CONTRAST_ENHANCEMENT  (all unused now)
# - SURYA_BATCH_SIZE, BATCH_WORKERS  (QARI leftovers, no longer relevant)
# """

# import os

# from config import cfg

# _ocr = cfg.ocr

# # -----------------------------
# # Language
# # -----------------------------
# OCR_LANGUAGE = _ocr.get("language", "ar")

# # -----------------------------
# # Engine
# # -----------------------------
# _engine = _ocr.get("engine", {})
# OCR_PROVIDER = _engine.get("provider", "gcv")

# # -----------------------------
# # Google Cloud Vision
# # -----------------------------
# _gcv = _ocr.get("gcv", {})
# GCV_FEATURE = _gcv.get("feature", "DOCUMENT_TEXT_DETECTION")

# # Credentials: prefer explicit file path from settings; fall back to the
# # standard GOOGLE_APPLICATION_CREDENTIALS env var (which is also what the
# # google-auth library reads automatically).  For Workload Identity on
# # GKE / Cloud Run, leave both blank — the client library handles it.
# GCV_CREDENTIALS_FILE: str = (
#     _gcv.get("credentials_file") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
# )
# if GCV_CREDENTIALS_FILE:
#     os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GCV_CREDENTIALS_FILE)

# # -----------------------------
# # LLM Refinement
# # -----------------------------
# _refinement = _ocr.get("refinement", {})
# REFINEMENT_ENABLED: bool = _refinement.get("enabled", True)
# REFINEMENT_TIMEOUT: int = _refinement.get("timeout_seconds", 60)

# # -----------------------------
# # Confidence thresholds
# # -----------------------------
# _confidence = _ocr.get("confidence", {})
# HIGH_CONFIDENCE_THRESHOLD: float = _confidence.get("high_threshold", 0.90)
# MEDIUM_CONFIDENCE_THRESHOLD: float = _confidence.get("medium_threshold", 0.65)

# # -----------------------------
# # Security / ingestion
# # -----------------------------
# _security = _ocr.get("security", {})
# MAX_FILE_SIZE_MB: int = _security.get("max_file_size_mb", 50)
# ALLOWED_EXTENSIONS: list = _security.get(
#     "allowed_extensions",
#     [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf", ".webp"],
# )

# # -----------------------------
# # Paths
# # -----------------------------
# import os as _os
# _OCR_DIR = _os.path.normpath(
#     _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "OCR")
# )
# BASE_DIR = _OCR_DIR
# DICTIONARY_PATH = _os.path.join(_OCR_DIR, "dictionaries", "legal_arabic.txt")



"""
config.ocr
----------
OCR pipeline constants sourced from ``settings.yaml`` via the centralised
``config`` module.

Removed from previous version
------------------------------
- PREPROCESSING_ENABLED  (preprocessing stage removed — GCV handles it)
- ENABLE_RESOLUTION_CHECK, MIN_DPI, ENABLE_DESKEW, ENABLE_DENOISE,
  ENABLE_BORDER_REMOVAL, ENABLE_CONTRAST_ENHANCEMENT  (all unused now)
- SURYA_BATCH_SIZE, BATCH_WORKERS  (QARI leftovers, no longer relevant)

Added in this version
---------------------
- REFINEMENT_CONFIDENCE_THRESHOLD  — pages at or above this GCV confidence
  score skip LLM refinement entirely.  Printed Arabic legal text is typically
  0.93–0.97; handwritten text 0.70–0.85.  Default 0.92 skips printed pages
  and keeps refinement only for handwritten or degraded-quality pages.

- MAX_WORKERS  — number of threads in the ThreadPoolExecutor that processes
  pages concurrently.  Both the GCV call and the LLM call are I/O-bound
  (network), so threads give near-linear speedup.  Default 4 covers typical
  document sizes without overwhelming either API's rate limits.

- REFINEMENT_TIMEOUT  — reduced default from 60 s to 30 s to match the
  faster ``gemini-2.5-flash-lite`` model used for OCR refinement.  Increase
  this if you switch back to the thinking model (``"high"`` tier).
"""

import os

from config import cfg

_ocr = cfg.ocr

# -----------------------------
# Language
# -----------------------------
OCR_LANGUAGE = _ocr.get("language", "ar")

# -----------------------------
# Engine
# -----------------------------
_engine = _ocr.get("engine", {})
OCR_PROVIDER = _engine.get("provider", "gcv")

# -----------------------------
# Google Cloud Vision
# -----------------------------
_gcv = _ocr.get("gcv", {})
GCV_FEATURE = _gcv.get("feature", "DOCUMENT_TEXT_DETECTION")

# Credentials: prefer explicit file path from settings; fall back to the
# standard GOOGLE_APPLICATION_CREDENTIALS env var (which is also what the
# google-auth library reads automatically).  For Workload Identity on
# GKE / Cloud Run, leave both blank — the client library handles it.
GCV_CREDENTIALS_FILE: str = (
    _gcv.get("credentials_file") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
)
if GCV_CREDENTIALS_FILE:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GCV_CREDENTIALS_FILE)

# -----------------------------
# LLM Refinement
# -----------------------------
_refinement = _ocr.get("refinement", {})
REFINEMENT_ENABLED: bool = _refinement.get("enabled", True)

# Timeout reduced from 60 s → 30 s to match gemini-2.5-flash-lite latency.
# If you switch llm_tier back to "high" (thinking model), raise this to 90 s.
REFINEMENT_TIMEOUT: int = _refinement.get("timeout_seconds", 30)

# Pages whose GCV word-confidence mean is at or above this value skip LLM
# refinement.  Printed text: typically 0.93–0.97.  Handwriting: 0.70–0.85.
# Setting this to 1.01 effectively disables the skip (always refine).
REFINEMENT_CONFIDENCE_THRESHOLD: float = _refinement.get(
    "confidence_threshold", 0.92
)

# -----------------------------
# Confidence thresholds
# -----------------------------
_confidence = _ocr.get("confidence", {})
HIGH_CONFIDENCE_THRESHOLD: float = _confidence.get("high_threshold", 0.90)
MEDIUM_CONFIDENCE_THRESHOLD: float = _confidence.get("medium_threshold", 0.65)

# -----------------------------
# Security / ingestion
# -----------------------------
_security = _ocr.get("security", {})
MAX_FILE_SIZE_MB: int = _security.get("max_file_size_mb", 50)
ALLOWED_EXTENSIONS: list = _security.get(
    "allowed_extensions",
    [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf", ".webp"],
)

# -----------------------------
# Performance
# -----------------------------
_perf = _ocr.get("performance", {})
# Thread-pool size for concurrent page processing.
# Both GCV and LLM calls are network I/O — threads give real parallelism here.
# Default 4 is a safe ceiling for typical document sizes without hitting rate limits.
MAX_WORKERS: int = _perf.get("max_workers", 4)

# -----------------------------
# Paths
# -----------------------------
import os as _os

_OCR_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "OCR")
)
BASE_DIR = _OCR_DIR
DICTIONARY_PATH = _os.path.join(_OCR_DIR, "dictionaries", "legal_arabic.txt")