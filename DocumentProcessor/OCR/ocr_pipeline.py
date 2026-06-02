# """
# DocumentProcessor.OCR.ocr_pipeline
# ------------------------------------
# Main OCR orchestrator.  ``run_ocr`` ties every pipeline stage together.

# Pipeline stages (per page)
# --------------------------
# 1. Ingest   — PDF/image → List[PIL.Image]          (ingestion.py)
# 2. Guard    — resize if image exceeds GCV 20MB limit (inline)
# 3. OCR      — GCV DOCUMENT_TEXT_DETECTION           (gcv_engine.py)
# 4. Refine   — LLM Classical Arabic correction       (llm_refinement.py) [toggleable]

# Removed stages (previously used for QARI, unnecessary for GCV)
# --------------------------------------------------------------
# - restore_image  (CLAHE contrast enhancement)
# - perspective_correct (contour / adaptive-threshold deskew)
# - normalize_numerals  (Arabic-Indic numeral → ASCII)

# GCV handles skew, lighting, and perspective internally and more accurately
# than the old CV preprocessing.  Numeral normalisation added no measurable
# accuracy gain over raw GCV output and is skipped.

# The only preprocessing kept is a simple pixel-dimension cap so images from
# high-resolution phone cameras (12MP+) stay well under GCV's 20 MB limit.

# Config keys
# -----------
# All keys live in ``settings.yaml`` under ``ocr`` and are exposed via
# ``config.ocr``.  Any key can be overridden at call-time via the ``config``
# dict argument to ``run_ocr``.

# Toggle for A/B testing:
#     refine_enabled   bool  default True  — LLM refinement on/off
# """

# from __future__ import annotations

# import io
# import logging
# import time
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Optional

# from PIL import Image

# from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
# from DocumentProcessor.OCR.ingestion import ingest_document
# from DocumentProcessor.OCR.llm_refinement import refine_ocr_text
# from DocumentProcessor.OCR.models import OCRDocumentResult, OCRPageResult

# logger = logging.getLogger(__name__)

# # GCV hard limit is 20 MB per image.  We target 15 MB to leave headroom.
# _GCV_MAX_BYTES = 15 * 1024 * 1024


# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------

# def run_ocr(
#     file_path: str,
#     doc_id: Optional[str] = None,
#     config: Optional[dict] = None,
# ) -> OCRDocumentResult:
#     """Run the full OCR pipeline on a PDF or image document.

#     Parameters
#     ----------
#     file_path:
#         Absolute path to the input file (PDF or image).
#     doc_id:
#         Optional identifier stored in result metadata.
#     config:
#         Optional dict of overrides merged on top of settings.yaml defaults.
#         Useful for A/B testing refinement::

#             run_ocr(path, config={"refine_enabled": False})

#     Returns
#     -------
#     OCRDocumentResult
#     """
#     cfg = _resolve_config(config)
#     t0 = time.time()

#     logger.info(
#         "run_ocr start: file=%s refine=%s",
#         file_path,
#         cfg["refine_enabled"],
#     )

#     pages = ingest_document(
#         file_path,
#         pdf_dpi=cfg["pdf_dpi"],
#         max_file_size_mb=cfg["max_file_size_mb"],
#         allowed_extensions=cfg["allowed_extensions"],
#     )
#     logger.info("Ingested %d page(s) from %s", len(pages), Path(file_path).name)

#     engine = get_gcv_engine()
#     page_results: list[OCRPageResult] = []

#     for i, page_img in enumerate(pages):
#         page_num = i + 1
#         t0_page = time.time()
#         logger.info("Processing page %d / %d …", page_num, len(pages))

#         try:
#             # ---- Size guard (only preprocessing kept for GCV) -------------
#             page_img = _enforce_size_limit(page_img, page_num)

#             # ---- GCV OCR --------------------------------------------------
#             ocr_result = engine.ocr_page(page_img, page_number=page_num)
#             raw_text: str = ocr_result.get("raw_text", "")
#             error: Optional[str] = ocr_result.get("error")

#             # ---- LLM refinement ------------------------------------------
#             # Skipped when OCR errored — never send empty/garbage to the LLM.
#             if cfg["refine_enabled"] and not bool(error):
#                 refined_text = refine_ocr_text(
#                     raw_text=raw_text,
#                     page_number=page_num,
#                     timeout=cfg["refine_timeout"],
#                 )
#             else:
#                 refined_text = ""

#             elapsed = time.time() - t0_page
#             logger.info(
#                 "Page %d done in %.2fs | conf=%.4f | refined=%s",
#                 page_num,
#                 elapsed,
#                 ocr_result.get("confidence") or 0.0,
#                 cfg["refine_enabled"] and not bool(error),
#             )

#             page_results.append(
#                 OCRPageResult(
#                     page_number=page_num,
#                     raw_text=raw_text,
#                     normalized_text=raw_text,   # kept for schema compat; same as raw
#                     refined_text=refined_text,
#                     perspective_corrected=False, # no longer applied
#                     confidence=ocr_result.get("confidence"),
#                     word_confidences=ocr_result.get("word_confidences"),
#                     error=error,
#                 )
#             )

#         except Exception as exc:
#             logger.exception("Unhandled error on page %d: %s", page_num, exc)
#             page_results.append(OCRPageResult(page_number=page_num, error=str(exc)))

#     total_elapsed = time.time() - t0
#     logger.info(
#         "run_ocr complete: %d page(s) in %.2fs", len(page_results), total_elapsed
#     )

#     metadata = {
#         "filename": Path(file_path).name,
#         "doc_id": doc_id,
#         "total_pages": len(page_results),
#         "model_used": "google-cloud-vision",
#         "gcv_feature": cfg["gcv_feature"],
#         "timestamp": datetime.now(timezone.utc).isoformat(),
#         "processing_time_seconds": round(total_elapsed, 2),
#         "refine_enabled": cfg["refine_enabled"],
#     }

#     return OCRDocumentResult(metadata=metadata, pages=page_results)


# # ---------------------------------------------------------------------------
# # Size guard
# # ---------------------------------------------------------------------------

# def _enforce_size_limit(
#     image: Image.Image,
#     page_num: int,
#     max_bytes: int = _GCV_MAX_BYTES,
# ) -> Image.Image:
#     """Shrink image if its PNG encoding would exceed GCV's 20 MB limit.

#     Iteratively scales down by 10 % until the encoded size is safe.
#     In practice this only triggers for very high-resolution phone photos
#     (≥ 12 MP at high quality).  Most scans and normal photos pass through
#     untouched.
#     """
#     buf = io.BytesIO()
#     img = image.convert("RGB") if image.mode != "RGB" else image
#     img.save(buf, format="PNG")
#     size = buf.tell()

#     if size <= max_bytes:
#         return img

#     logger.info(
#         "Page %d: encoded size %.1f MB exceeds limit — resizing …",
#         page_num,
#         size / (1024 * 1024),
#     )

#     scale = 0.9
#     while size > max_bytes and min(img.size) > 100:
#         w = int(img.width * scale)
#         h = int(img.height * scale)
#         img = img.resize((w, h), Image.LANCZOS)
#         buf = io.BytesIO()
#         img.save(buf, format="PNG")
#         size = buf.tell()

#     logger.info(
#         "Page %d: resized to %dx%d (%.1f MB)",
#         page_num, img.width, img.height, size / (1024 * 1024),
#     )
#     return img


# # ---------------------------------------------------------------------------
# # Config resolution
# # ---------------------------------------------------------------------------

# def _resolve_config(config: Optional[dict] = None) -> dict:
#     """Build a complete config dict from settings.yaml defaults + overrides."""
#     from config.ocr import (  # noqa: PLC0415
#         ALLOWED_EXTENSIONS,
#         GCV_FEATURE,
#         HIGH_CONFIDENCE_THRESHOLD,
#         MAX_FILE_SIZE_MB,
#         MEDIUM_CONFIDENCE_THRESHOLD,
#         REFINEMENT_ENABLED,
#         REFINEMENT_TIMEOUT,
#     )

#     defaults: dict = {
#         # Engine
#         "gcv_feature": GCV_FEATURE,
#         # Ingestion
#         "pdf_dpi": 300,
#         "max_file_size_mb": MAX_FILE_SIZE_MB,
#         "allowed_extensions": ALLOWED_EXTENSIONS,
#         # Confidence thresholds (used by gcv_engine._band)
#         "high_threshold": HIGH_CONFIDENCE_THRESHOLD,
#         "medium_threshold": MEDIUM_CONFIDENCE_THRESHOLD,
#         # LLM refinement
#         "refine_enabled": REFINEMENT_ENABLED,
#         "refine_timeout": REFINEMENT_TIMEOUT,
#     }

#     if config:
#         defaults.update(config)

#     return defaults


"""
DocumentProcessor.OCR.ocr_pipeline
------------------------------------
Main OCR orchestrator.  ``run_ocr`` ties every pipeline stage together.

Pipeline stages (per page)
--------------------------
1. Ingest   — PDF/image → List[PIL.Image]          (ingestion.py)
2. OCR      — GCV DOCUMENT_TEXT_DETECTION           (gcv_engine.py)
3. Refine   — LLM Classical Arabic correction       (llm_refinement.py) [toggleable]

Removed from this version
--------------------------
- ``_enforce_size_limit`` — the PNG encode it performed to measure image size
  was itself a major source of latency (one full PNG encode per page *before*
  the GCV call, plus a second one *inside* the GCV call).  Size limiting is
  now handled transparently inside ``GCVEngine._pil_to_bytes`` using JPEG,
  which is 10–15× smaller for the same image and never needs resizing in
  practice at 300 DPI.

Performance
-----------
Pages are processed **concurrently** via ``ThreadPoolExecutor``.  Both the
GCV call and the LLM call are I/O-bound (network), so Python threads give
near-linear speedup up to the point where one of the upstream APIs becomes
the bottleneck.

For a 4-page document the wall-clock time goes from
``sum(page times)`` → ``max(page times)``, i.e. from ~285 s to ~20 s once
all individual-page fixes are in place.

LLM refinement is additionally **skipped automatically** for pages whose GCV
confidence meets or exceeds ``refine_confidence_threshold`` (default 0.92).
Printed Arabic legal text consistently scores 0.93–0.97; handwritten text
typically scores 0.70–0.85.  The threshold is tunable in ``settings.yaml``::

    ocr:
      refinement:
        confidence_threshold: 0.92

Config keys
-----------
All keys live in ``settings.yaml`` under ``ocr`` and are exposed via
``config.ocr``.  Any key can be overridden at call-time via the ``config``
dict argument to ``run_ocr``.

Toggle for A/B testing:
    refine_enabled                 bool   default True  — LLM refinement on/off
    refine_confidence_threshold    float  default 0.92  — skip threshold
    max_workers                    int    default 4     — thread-pool size
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PIL import Image

from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
from DocumentProcessor.OCR.ingestion import ingest_document
from DocumentProcessor.OCR.llm_refinement import refine_ocr_text
from DocumentProcessor.OCR.models import OCRDocumentResult, OCRPageResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ocr(
    file_path: str,
    doc_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> OCRDocumentResult:
    """Run the full OCR pipeline on a PDF or image document.

    Parameters
    ----------
    file_path:
        Absolute path to the input file (PDF or image).
    doc_id:
        Optional identifier stored in result metadata.
    config:
        Optional dict of overrides merged on top of settings.yaml defaults.
        Examples::

            # Disable LLM refinement entirely
            run_ocr(path, config={"refine_enabled": False})

            # Lower the confidence skip threshold (refine more pages)
            run_ocr(path, config={"refine_confidence_threshold": 0.80})

            # Single-threaded (useful for debugging log order)
            run_ocr(path, config={"max_workers": 1})

    Returns
    -------
    OCRDocumentResult
    """
    cfg = _resolve_config(config)
    t0 = time.time()

    logger.info(
        "run_ocr start: file=%s refine=%s threshold=%.2f workers=%d",
        file_path,
        cfg["refine_enabled"],
        cfg["refine_confidence_threshold"],
        cfg["max_workers"],
    )

    pages = ingest_document(
        file_path,
        pdf_dpi=cfg["pdf_dpi"],
        max_file_size_mb=cfg["max_file_size_mb"],
        allowed_extensions=cfg["allowed_extensions"],
    )
    n_pages = len(pages)
    logger.info("Ingested %d page(s) from %s", n_pages, Path(file_path).name)

    engine = get_gcv_engine()
    max_workers = min(cfg["max_workers"], n_pages)

    # Pre-allocate result list so pages are always returned in document order,
    # regardless of which thread finishes first.
    page_results: list[Optional[OCRPageResult]] = [None] * n_pages

    logger.info("Processing %d page(s) across %d worker(s) …", n_pages, max_workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_process_page, i, img, n_pages, engine, cfg): i
            for i, img in enumerate(pages)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            page_results[idx] = future.result()

    total_elapsed = time.time() - t0
    logger.info(
        "run_ocr complete: %d page(s) in %.2fs", n_pages, total_elapsed,
    )

    metadata = {
        "filename": Path(file_path).name,
        "doc_id": doc_id,
        "total_pages": n_pages,
        "model_used": "google-cloud-vision",
        "gcv_feature": cfg["gcv_feature"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": round(total_elapsed, 2),
        "refine_enabled": cfg["refine_enabled"],
        "refine_confidence_threshold": cfg["refine_confidence_threshold"],
        "max_workers": max_workers,
    }

    return OCRDocumentResult(metadata=metadata, pages=page_results)


# ---------------------------------------------------------------------------
# Per-page worker (runs inside thread-pool)
# ---------------------------------------------------------------------------

def _process_page(
    idx: int,
    page_img: Image.Image,
    n_pages: int,
    engine,
    cfg: dict,
) -> OCRPageResult:
    """Process a single page: GCV OCR → optional LLM refinement.

    Designed to run concurrently in a ``ThreadPoolExecutor``.
    Both ``GCVEngine`` and the LangChain LLM client are stateless per-call
    and safe to share across threads.

    Parameters
    ----------
    idx:
        0-based page index (used to reconstruct document order).
    page_img:
        PIL Image for this page, already ingested.
    n_pages:
        Total pages in the document (used for progress logging only).
    engine:
        The shared ``GCVEngine`` singleton.
    cfg:
        Resolved config dict from ``_resolve_config``.

    Returns
    -------
    OCRPageResult
    """
    page_num = idx + 1
    t0_page = time.time()
    logger.info("Processing page %d / %d …", page_num, n_pages)

    try:
        # ---- GCV OCR -------------------------------------------------------
        # _pil_to_bytes inside engine.ocr_page now handles JPEG encoding and
        # size limiting — no separate preprocessing step needed.
        ocr_result = engine.ocr_page(page_img, page_number=page_num)
        raw_text: str = ocr_result.get("raw_text", "")
        error: Optional[str] = ocr_result.get("error")
        confidence: Optional[float] = ocr_result.get("confidence")

        # ---- LLM refinement (confidence-gated) -----------------------------
        # Skip refinement when:
        #   a) globally disabled via refine_enabled=False
        #   b) GCV errored — never send empty/garbage text to the LLM
        #   c) GCV confidence is at or above the threshold — this page is
        #      printed text that GCV already reads accurately; the LLM would
        #      make zero meaningful changes and costs 8–50 s for nothing
        refine_threshold: float = cfg["refine_confidence_threshold"]
        above_threshold: bool = (
            confidence is not None and confidence >= refine_threshold
        )
        should_refine: bool = (
            cfg["refine_enabled"]
            and not bool(error)
            and not above_threshold
        )

        if should_refine:
            refined_text = refine_ocr_text(
                raw_text=raw_text,
                page_number=page_num,
                timeout=cfg["refine_timeout"],
            )
        else:
            refined_text = ""
            if above_threshold and cfg["refine_enabled"]:
                logger.info(
                    "Page %d: LLM skipped — conf=%.4f ≥ threshold=%.2f",
                    page_num,
                    confidence,
                    refine_threshold,
                )

        elapsed = time.time() - t0_page
        logger.info(
            "Page %d done in %.2fs | conf=%s | refined=%s",
            page_num,
            elapsed,
            f"{confidence:.4f}" if confidence is not None else "N/A",
            should_refine,
        )

        return OCRPageResult(
            page_number=page_num,
            raw_text=raw_text,
            normalized_text=raw_text,  # kept for schema compat; same as raw
            refined_text=refined_text,
            perspective_corrected=False,
            confidence=confidence,
            word_confidences=ocr_result.get("word_confidences"),
            error=error,
        )

    except Exception as exc:
        elapsed = time.time() - t0_page
        logger.exception(
            "Unhandled error on page %d after %.2fs: %s", page_num, elapsed, exc
        )
        return OCRPageResult(page_number=page_num, error=str(exc))


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_config(config: Optional[dict] = None) -> dict:
    """Build a complete config dict from settings.yaml defaults + overrides."""
    from config.ocr import (  # noqa: PLC0415
        ALLOWED_EXTENSIONS,
        GCV_FEATURE,
        HIGH_CONFIDENCE_THRESHOLD,
        MAX_FILE_SIZE_MB,
        MAX_WORKERS,
        MEDIUM_CONFIDENCE_THRESHOLD,
        REFINEMENT_CONFIDENCE_THRESHOLD,
        REFINEMENT_ENABLED,
        REFINEMENT_TIMEOUT,
    )

    defaults: dict = {
        # Engine
        "gcv_feature": GCV_FEATURE,
        # Ingestion
        "pdf_dpi": 300,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "allowed_extensions": ALLOWED_EXTENSIONS,
        # Confidence thresholds (used by gcv_engine._band)
        "high_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "medium_threshold": MEDIUM_CONFIDENCE_THRESHOLD,
        # LLM refinement
        "refine_enabled": REFINEMENT_ENABLED,
        "refine_timeout": REFINEMENT_TIMEOUT,
        "refine_confidence_threshold": REFINEMENT_CONFIDENCE_THRESHOLD,
        # Parallelism
        "max_workers": MAX_WORKERS,
    }

    if config:
        defaults.update(config)

    return defaults