"""
config.ocr
----------
OCR pipeline constants sourced from ``settings.yaml`` via the centralised
``config`` module.

All new GCV / LLM-refinement constants are defined here so every part of
the OCR package reads from a single source of truth.
"""

import os

from config import cfg

_ocr = cfg.ocr

# -----------------------------
# Language
# -----------------------------
OCR_LANGUAGE = _ocr.get("language", "ar")
USE_GPU = _ocr.get("use_gpu", False)

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
    # Make the env var consistent for the google-auth library
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", GCV_CREDENTIALS_FILE)

# -----------------------------
# LLM Refinement
# -----------------------------
_refinement = _ocr.get("refinement", {})
REFINEMENT_ENABLED: bool = _refinement.get("enabled", True)
REFINEMENT_TIMEOUT: int = _refinement.get("timeout_seconds", 60)
PREPROCESSING_ENABLED: bool = _refinement.get("preprocessing_enabled", True)

# -----------------------------
# Confidence thresholds
# -----------------------------
_confidence = _ocr.get("confidence", {})
HIGH_CONFIDENCE_THRESHOLD: float = _confidence.get("high_threshold", 0.90)
MEDIUM_CONFIDENCE_THRESHOLD: float = _confidence.get("medium_threshold", 0.65)

# -----------------------------
# Preprocessing
# -----------------------------
_preproc = _ocr.get("preprocessing", {})
ENABLE_RESOLUTION_CHECK: bool = _preproc.get("enable_resolution_check", True)
MIN_DPI: int = _preproc.get("min_dpi", 150)
ENABLE_DESKEW: bool = _preproc.get("enable_deskew", True)
ENABLE_DENOISE: bool = _preproc.get("enable_denoise", False)
ENABLE_BORDER_REMOVAL: bool = _preproc.get("enable_border_removal", True)
ENABLE_CONTRAST_ENHANCEMENT: bool = _preproc.get("enable_contrast_enhancement", True)

# -----------------------------
# Post-processing
# -----------------------------
_postproc = _ocr.get("postprocessing", {})
ENABLE_DICTIONARY_CORRECTION: bool = _postproc.get("enable_dictionary_correction", True)
MAX_LEVENSHTEIN_DISTANCE: int = _postproc.get("max_levenshtein_distance", 2)
NORMALIZE_DIGITS: str = _postproc.get("normalize_digits", "arabic_indic")

# -----------------------------
# Security
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
SURYA_BATCH_SIZE: int = _perf.get("surya_batch_size", 4)
BATCH_WORKERS: int = _perf.get("batch_workers", 4)

# -----------------------------
# Paths
# -----------------------------
_OCR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "OCR")
)
BASE_DIR = _OCR_DIR
DICTIONARY_PATH = os.path.join(_OCR_DIR, "dictionaries", "legal_arabic.txt")
