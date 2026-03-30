"""
config.ocr
----------
OCR pipeline constants.

Moved from ``OCR/config.py`` during config consolidation.
All values are sourced from the centralized ``config`` module.
"""

import os

from config import cfg

_ocr = cfg.ocr

# -----------------------------
# Engine
# -----------------------------
OCR_LANGUAGE = _ocr.get("language", "ar")
USE_GPU = _ocr.get("use_gpu", True)

# -----------------------------
# Preprocessing
# -----------------------------
_preproc = _ocr.get("preprocessing", {})
ENABLE_RESOLUTION_CHECK = _preproc.get("enable_resolution_check", True)
MIN_DPI = _preproc.get("min_dpi", 150)
ENABLE_DESKEW = _preproc.get("enable_deskew", True)
ENABLE_DENOISE = _preproc.get("enable_denoise", False)
ENABLE_BORDER_REMOVAL = _preproc.get("enable_border_removal", True)
ENABLE_CONTRAST_ENHANCEMENT = _preproc.get("enable_contrast_enhancement", True)

# -----------------------------
# Confidence Thresholds
# -----------------------------
_confidence = _ocr.get("confidence", {})
HIGH_CONFIDENCE_THRESHOLD = _confidence.get("high_threshold", 0.85)
MEDIUM_CONFIDENCE_THRESHOLD = _confidence.get("medium_threshold", 0.60)

# -----------------------------
# Post-processing
# -----------------------------
_postproc = _ocr.get("postprocessing", {})
ENABLE_DICTIONARY_CORRECTION = _postproc.get("enable_dictionary_correction", True)
MAX_LEVENSHTEIN_DISTANCE = _postproc.get("max_levenshtein_distance", 2)
NORMALIZE_DIGITS = _postproc.get("normalize_digits", "arabic_indic")

# -----------------------------
# Security
# -----------------------------
_security = _ocr.get("security", {})
MAX_FILE_SIZE_MB = _security.get("max_file_size_mb", 50)
ALLOWED_EXTENSIONS = _security.get("allowed_extensions", [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"])

# -----------------------------
# Performance
# -----------------------------
_perf = _ocr.get("performance", {})
SURYA_BATCH_SIZE = _perf.get("surya_batch_size", 4)
BATCH_WORKERS = _perf.get("batch_workers", 4)

# -----------------------------
# Paths -- resolve relative to the OCR package directory
# -----------------------------
_OCR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "OCR")
_OCR_DIR = os.path.normpath(_OCR_DIR)
BASE_DIR = _OCR_DIR
DICTIONARY_PATH = os.path.join(_OCR_DIR, "dictionaries", "legal_arabic.txt")
