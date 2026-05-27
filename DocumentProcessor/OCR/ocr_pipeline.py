"""
DocumentProcessor.OCR.ocr_pipeline
------------------------------------
Main OCR orchestrator.  ``run_ocr`` ties every pipeline stage together.

Pipeline stages (per page)
--------------------------
1. Ingest      — PDF/image → List[PIL.Image]       (ingestion.py)
2. Restore     — resize + CLAHE contrast           (restoration.py)    [toggleable]
3. Deskew      — perspective / contour correction  (perspective_correction.py) [toggleable]
4. OCR         — GCV DOCUMENT_TEXT_DETECTION       (gcv_engine.py)
5. Normalise   — Arabic-Indic numeral → ASCII      (text_reconstruction.py)
6. Refine      — LLM Classical Arabic correction   (llm_refinement.py) [toggleable]

Config keys
-----------
All keys live in ``settings.yaml`` under the ``ocr`` section and are exposed
via ``config.ocr``.  Any key can be overridden at call-time by passing a
``config`` dict to ``run_ocr``.

Key toggles for A/B testing:
    preprocessing_enabled  bool  default True   — stages 2 & 3
    refine_enabled         bool  default True   — stage 6
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
from DocumentProcessor.OCR.ingestion import ingest_document
from DocumentProcessor.OCR.llm_refinement import refine_ocr_text
from DocumentProcessor.OCR.models import OCRDocumentResult, OCRPageResult
from DocumentProcessor.OCR.perspective_correction import perspective_correct
from DocumentProcessor.OCR.restoration import restore_image
from DocumentProcessor.OCR.text_reconstruction import normalize_numerals

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
        Absolute path to the input file.
    doc_id:
        Optional identifier stored in the result metadata.
    config:
        Optional dict of config overrides.  Merged on top of defaults
        derived from ``settings.yaml``.  Useful for A/B testing::

            run_ocr(path, config={"preprocessing_enabled": False})
            run_ocr(path, config={"refine_enabled": False})

    Returns
    -------
    OCRDocumentResult
    """
    cfg = _resolve_config(config)
    t0 = time.time()

    logger.info(
        "run_ocr start: file=%s preprocessing=%s refine=%s",
        file_path,
        cfg["preprocessing_enabled"],
        cfg["refine_enabled"],
    )

    pages = ingest_document(
        file_path,
        pdf_dpi=cfg["pdf_dpi"],
        max_file_size_mb=cfg["max_file_size_mb"],
        allowed_extensions=cfg["allowed_extensions"],
    )
    logger.info("Ingested %d page(s) from %s", len(pages), Path(file_path).name)

    engine = get_gcv_engine()
    page_results: list[OCRPageResult] = []

    for i, page_img in enumerate(pages):
        page_num = i + 1
        t0_page = time.time()
        logger.info("Processing page %d / %d …", page_num, len(pages))

        try:
            page_img, was_corrected = _preprocess(page_img, cfg, page_num)

            # ---- GCV OCR --------------------------------------------------
            ocr_result = engine.ocr_page(page_img, page_number=page_num)
            raw_text: str = ocr_result.get("raw_text", "")
            error: Optional[str] = ocr_result.get("error")

            # ---- Numeral normalisation ------------------------------------
            normalized_text = normalize_numerals(raw_text) if raw_text else ""

            # ---- LLM refinement ------------------------------------------
            # Refinement is skipped automatically when the OCR stage errored
            # so we never send empty/garbage text to the LLM.
            if cfg["refine_enabled"] and not bool(error):
                refined_text = refine_ocr_text(
                    raw_text=normalized_text,
                    page_number=page_num,
                    timeout=cfg["refine_timeout"],
                )
            else:
                refined_text = ""

            elapsed = time.time() - t0_page
            logger.info(
                "Page %d done in %.2fs | conf=%.4f | corrected=%s | refined=%s",
                page_num,
                elapsed,
                ocr_result.get("confidence") or 0.0,
                was_corrected,
                cfg["refine_enabled"] and not bool(error),
            )

            page_results.append(
                OCRPageResult(
                    page_number=page_num,
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    refined_text=refined_text,
                    perspective_corrected=was_corrected,
                    confidence=ocr_result.get("confidence"),
                    word_confidences=ocr_result.get("word_confidences"),
                    error=error,
                )
            )

        except Exception as exc:
            logger.exception("Unhandled error on page %d: %s", page_num, exc)
            page_results.append(OCRPageResult(page_number=page_num, error=str(exc)))

    total_elapsed = time.time() - t0
    logger.info(
        "run_ocr complete: %d page(s) in %.2fs", len(page_results), total_elapsed
    )

    metadata = {
        "filename": Path(file_path).name,
        "doc_id": doc_id,
        "total_pages": len(page_results),
        "model_used": "google-cloud-vision",
        "gcv_feature": cfg["gcv_feature"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": round(total_elapsed, 2),
        "perspective_corrected": any(p.perspective_corrected for p in page_results),
        "preprocessing_enabled": cfg["preprocessing_enabled"],
        "refine_enabled": cfg["refine_enabled"],
    }

    return OCRDocumentResult(metadata=metadata, pages=page_results)


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def _preprocess(page_img, cfg: dict, page_num: int):
    """Run restore + perspective correction if preprocessing is enabled.

    Returns (processed_image, was_perspective_corrected).
    """
    if not cfg["preprocessing_enabled"]:
        logger.debug("Preprocessing disabled for page %d — raw image to GCV", page_num)
        return page_img, False

    restored = restore_image(
        page_img,
        max_image_dimension=cfg["max_image_dimension"],
        clahe_clip_limit=cfg["clahe_clip_limit"],
        clahe_tile_grid_size=tuple(cfg["clahe_tile_grid_size"]),
    )

    corrected, was_corrected = perspective_correct(
        restored,
        min_area_ratio=cfg["min_area_ratio"],
        canny_low=cfg["canny_low"],
        canny_high=cfg["canny_high"],
        blur_kernel=tuple(cfg["blur_kernel"]),
        dilate_kernel=tuple(cfg["dilate_kernel"]),
        dilate_iterations=cfg["dilate_iterations"],
        top_n_contours=cfg["top_n_contours"],
        approx_epsilon_cd=cfg["approx_epsilon_cd"],
        expand_margin_cd=cfg["expand_margin_cd"],
        block_size=cfg["block_size"],
        c_constant=cfg["c_constant"],
        close_kernel=tuple(cfg["close_kernel"]),
        close_iterations=cfg["close_iterations"],
        open_kernel=tuple(cfg["open_kernel"]),
        open_iterations=cfg["open_iterations"],
        approx_epsilon_at=cfg["approx_epsilon_at"],
        expand_margin_at=cfg["expand_margin_at"],
    )

    return corrected, was_corrected


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
        MEDIUM_CONFIDENCE_THRESHOLD,
        PREPROCESSING_ENABLED,
        REFINEMENT_ENABLED,
        REFINEMENT_TIMEOUT,
    )

    defaults: dict = {
        # Engine
        "gcv_feature": GCV_FEATURE,
        # Preprocessing toggle (stages: restore + perspective-correct)
        "preprocessing_enabled": PREPROCESSING_ENABLED,
        # Image preprocessing parameters
        "max_image_dimension": 4000,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": [8, 8],
        "pdf_dpi": 400,
        "min_area_ratio": 0.35,
        "canny_low": 50,
        "canny_high": 150,
        "blur_kernel": [5, 5],
        "dilate_kernel": [3, 3],
        "dilate_iterations": 1,
        "top_n_contours": 5,
        "approx_epsilon_cd": 0.02,
        "expand_margin_cd": 0.02,
        "block_size": 35,
        "c_constant": -10,
        "close_kernel": [30, 30],
        "close_iterations": 2,
        "open_kernel": [15, 15],
        "open_iterations": 1,
        "approx_epsilon_at": 0.02,
        "expand_margin_at": 0.02,
        # Confidence thresholds (used by gcv_engine._band)
        "high_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "medium_threshold": MEDIUM_CONFIDENCE_THRESHOLD,
        # LLM refinement
        "refine_enabled": REFINEMENT_ENABLED,
        "refine_timeout": REFINEMENT_TIMEOUT,
        # Security / ingestion
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "allowed_extensions": ALLOWED_EXTENSIONS,
    }

    if config:
        defaults.update(config)

    return defaults
