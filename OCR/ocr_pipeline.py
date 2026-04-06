"""
OCR.ocr_pipeline
----------------
Main orchestrator: ``process_document`` ties every pipeline stage together.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from OCR.ingestion import ingest_document
from OCR.models import OCRDocumentResult, OCRPageResult
from OCR.ocr_engine import get_engine
from OCR.perspective_correction import perspective_correct
from OCR.restoration import restore_image
from OCR.text_reconstruction import normalize_numerals

logger = logging.getLogger(__name__)

# Default OCR prompt
_DEFAULT_OCR_PROMPT = (
    "You are a strict OCR engine transcribing Arabic legal documents. "
    "Transcribe exactly what is written in this image. "
    "Do NOT correct spelling, grammar, punctuation, or any perceived mistakes. "
    "Do NOT add or remove any words. "
    "If something looks like a typo or error, transcribe it exactly as-is. "
    "Preserve all text exactly character by character."
)


def process_document(
    file_path: str,
    doc_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> OCRDocumentResult:
    """Run the full OCR pipeline on a document.

    Parameters
    ----------
    file_path:
        Path to the input file (PDF or image).
    doc_id:
        Optional document identifier for metadata.
    config:
        Optional configuration dict.  If ``None``, loads from
        ``config/ocr.py`` defaults.

    Returns
    -------
    OCRDocumentResult
        Structured result with metadata and per-page results.
    """
    cfg = _resolve_config(config)

    t0_total = time.time()

    # ── 1. Ingest ─────────────────────────────────────────────────────
    logger.info("Ingesting document: %s", file_path)
    pages = ingest_document(
        file_path,
        pdf_dpi=cfg["pdf_dpi"],
        max_file_size_mb=cfg["max_file_size_mb"],
        allowed_extensions=cfg["allowed_extensions"],
    )
    logger.info("Ingested %d page(s)", len(pages))

    # ── 2. Load engine (singleton) ────────────────────────────────────
    engine = get_engine(
        model_name=cfg["model_name"],
        max_new_tokens=cfg["max_new_tokens"],
        quantization=cfg["quantization"],
        torch_dtype_str=cfg["torch_dtype"],
        use_gpu=cfg["use_gpu"],
    )

    # ── 3-6. Process each page ────────────────────────────────────────
    page_results: list[OCRPageResult] = []

    for i, page_img in enumerate(pages):
        page_num = i + 1
        logger.info("Processing page %d/%d...", page_num, len(pages))
        t0_page = time.time()

        try:
            # 3. Restore
            restored = restore_image(
                page_img,
                max_image_dimension=cfg["max_image_dimension"],
                clahe_clip_limit=cfg["clahe_clip_limit"],
                clahe_tile_grid_size=tuple(cfg["clahe_tile_grid_size"]),
            )

            # 4. Perspective correct
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

            # 5. OCR with confidence
            ocr_result = engine.ocr_page(
                corrected,
                ocr_prompt=cfg["ocr_prompt"],
                page_number=page_num,
            )

            raw_text = ocr_result.get("raw_text", "")
            error = ocr_result.get("error")

            # 6. Text reconstruction
            normalized_text = normalize_numerals(raw_text) if raw_text else ""

            elapsed = time.time() - t0_page
            logger.info(
                "Page %d done in %.2fs (confidence=%.4f)",
                page_num,
                elapsed,
                ocr_result.get("confidence") or 0.0,
            )

            page_results.append(OCRPageResult(
                page_number=page_num,
                raw_text=raw_text,
                normalized_text=normalized_text,
                perspective_corrected=was_corrected,
                confidence=ocr_result.get("confidence"),
                word_confidences=ocr_result.get("word_confidences"),
                error=error,
            ))

        except Exception as exc:
            logger.exception("Error processing page %d: %s", page_num, exc)
            page_results.append(OCRPageResult(
                page_number=page_num,
                error=str(exc),
            ))

    total_elapsed = time.time() - t0_total
    logger.info("Document processed in %.2fs (%d pages)", total_elapsed, len(pages))

    # ── 7. Assemble result ────────────────────────────────────────────
    metadata = {
        "filename": Path(file_path).name,
        "doc_id": doc_id,
        "total_pages": len(page_results),
        "model_used": cfg["model_name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": round(total_elapsed, 2),
        "perspective_corrected": any(
            p.perspective_corrected for p in page_results
        ),
    }

    return OCRDocumentResult(metadata=metadata, pages=page_results)


def _resolve_config(config: Optional[dict] = None) -> dict:
    """Build a complete config dict with defaults, optionally overridden."""
    defaults = {
        # Model
        "model_name": "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
        "max_new_tokens": 4000,
        "quantization": "8bit",
        "torch_dtype": "float16",
        "use_gpu": True,
        # Preprocessing
        "max_image_dimension": 4000,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": [8, 8],
        "pdf_dpi": 400,
        # Perspective correction -- contour detection
        "min_area_ratio": 0.35,
        "canny_low": 50,
        "canny_high": 150,
        "blur_kernel": [5, 5],
        "dilate_kernel": [3, 3],
        "dilate_iterations": 1,
        "top_n_contours": 5,
        "approx_epsilon_cd": 0.02,
        "expand_margin_cd": 0.02,
        # Perspective correction -- adaptive threshold
        "block_size": 35,
        "c_constant": -10,
        "close_kernel": [30, 30],
        "close_iterations": 2,
        "open_kernel": [15, 15],
        "open_iterations": 1,
        "approx_epsilon_at": 0.02,
        "expand_margin_at": 0.02,
        # Confidence
        "high_threshold": 0.85,
        "medium_threshold": 0.60,
        # Post-processing
        "normalize_digits": True,
        # Security
        "max_file_size_mb": 50,
        "allowed_extensions": [
            ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf", ".webp",
        ],
        # Prompt
        "ocr_prompt": _DEFAULT_OCR_PROMPT,
    }

    if config is not None:
        defaults.update(config)

    return defaults
