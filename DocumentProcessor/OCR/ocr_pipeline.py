"""
DocumentProcessor.OCR.ocr_pipeline
------------------------------------
Main OCR orchestrator — v4.

Pipeline stages
---------------
Stage 1  Ingest     PDF/image → List[PIL.Image]             (ingestion.py)
Stage 2  GCV batch  All pages via chunked batch calls        (gcv_engine.py)
Stage 3  LLM refine Parallel refinement for low-conf pages  (llm_refinement.py)

LLM pre-warming
---------------
The LLM singleton (ChatGoogleGenerativeAI) takes ~18 s to initialise on first
use — importing the library, creating the client, establishing transport.
Previously this cost was paid *after* all GCV chunks finished, blocking the
pipeline for 18 s before any LLM call could begin.

Fix: a daemon thread is started at the very top of ``run_ocr``, before
ingestion, so the 18 s initialisation runs in parallel with:
  - PDF → image conversion   (~2 s)
  - GCV engine cold-start    (~8 s)
  - JPEG encoding            (~1 s)
  - GCV chunk 1              (~59 s for 16 pages at 200 DPI)
  - GCV chunk 2              (~22 s for 4 pages)

By the time chunks finish (~82 s later), the LLM has been ready for >60 s.
The ``_llm_prewarm_thread.join()`` call before the LLM stage is a no-op in
practice; it only blocks if something made initialisation unusually slow.

Expected improvement for a 20-page document: ~116 s → ~98 s.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from DocumentProcessor.OCR.gcv_engine import get_gcv_engine
from DocumentProcessor.OCR.ingestion import ingest_document
from DocumentProcessor.OCR.llm_refinement import prewarm_llm, refine_ocr_text
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
    """Run the full OCR pipeline on a PDF or image document."""
    cfg = _resolve_config(config)
    t0 = time.time()

    logger.info(
        "run_ocr start: file=%s dpi=%d refine=%s threshold=%.2f",
        file_path,
        cfg["pdf_dpi"],
        cfg["refine_enabled"],
        cfg["refine_confidence_threshold"],
    )

    # -----------------------------------------------------------------------
    # Pre-warm LLM singleton in background — runs during ingestion + GCV
    # so the ~18 s initialisation cost is invisible to the user.
    # -----------------------------------------------------------------------
    _llm_prewarm_thread: Optional[threading.Thread] = None
    if cfg["refine_enabled"]:
        _llm_prewarm_thread = threading.Thread(
            target=prewarm_llm,
            args=(cfg["refine_timeout"],),
            daemon=True,
            name="llm-prewarm",
        )
        _llm_prewarm_thread.start()
        logger.debug("LLM pre-warm thread started")

    # -----------------------------------------------------------------------
    # Stage 1: Ingest
    # -----------------------------------------------------------------------
    pages = ingest_document(
        file_path,
        pdf_dpi=cfg["pdf_dpi"],
        max_file_size_mb=cfg["max_file_size_mb"],
        allowed_extensions=cfg["allowed_extensions"],
    )
    n_pages = len(pages)
    logger.info("Ingested %d page(s) from %s", n_pages, Path(file_path).name)

    engine = get_gcv_engine()

    # -----------------------------------------------------------------------
    # Stage 2: GCV batch (chunked — max 16 images per call)
    # -----------------------------------------------------------------------
    t_gcv = time.time()
    gcv_results = engine.ocr_batch(pages)
    gcv_elapsed = time.time() - t_gcv
    logger.info(
        "GCV batch complete: %d page(s) in %.2fs (%.1fs/page avg)",
        n_pages, gcv_elapsed, gcv_elapsed / max(n_pages, 1),
    )

    for i, res in enumerate(gcv_results):
        conf = res.get("confidence")
        err  = res.get("error")
        logger.info(
            "Page %d GCV: conf=%s%s",
            i + 1,
            f"{conf:.4f}" if conf is not None else "N/A",
            f" ERROR={err}" if err else "",
        )

    # -----------------------------------------------------------------------
    # Stage 3: LLM refinement (parallel, confidence-gated)
    # -----------------------------------------------------------------------
    refine_threshold: float = cfg["refine_confidence_threshold"]

    def _needs_refine(result: dict) -> bool:
        if not cfg["refine_enabled"] or bool(result.get("error")):
            return False
        conf = result.get("confidence")
        return conf is None or conf < refine_threshold

    pages_to_refine = [i for i, r in enumerate(gcv_results) if _needs_refine(r)]
    skipped = [
        i for i, r in enumerate(gcv_results)
        if cfg["refine_enabled"] and not r.get("error") and not _needs_refine(r)
    ]

    for i in skipped:
        logger.info(
            "Page %d: LLM skipped — conf=%.4f ≥ threshold=%.2f",
            i + 1, gcv_results[i]["confidence"], refine_threshold,
        )

    refined_texts: dict[int, str] = {}

    if pages_to_refine:
        logger.info(
            "LLM refinement: %d page(s) to refine — %s",
            len(pages_to_refine), [p + 1 for p in pages_to_refine],
        )

        # Ensure the pre-warm thread has finished before we dispatch LLM calls.
        # In practice it completed ~60 s ago for any document > 4 pages.
        if _llm_prewarm_thread and _llm_prewarm_thread.is_alive():
            logger.debug("Waiting for LLM pre-warm thread to finish …")
            _llm_prewarm_thread.join()

        def _refine_page(page_idx: int) -> tuple[int, str]:
            text = refine_ocr_text(
                raw_text=gcv_results[page_idx]["raw_text"],
                page_number=page_idx + 1,
                timeout=cfg["refine_timeout"],
            )
            return page_idx, text

        llm_workers = min(len(pages_to_refine), cfg["max_workers"])
        t_llm = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=llm_workers) as pool:
            for page_idx, refined in pool.map(_refine_page, pages_to_refine):
                refined_texts[page_idx] = refined
        logger.info(
            "LLM refinement complete: %d page(s) in %.2fs",
            len(pages_to_refine), time.time() - t_llm,
        )

    # -----------------------------------------------------------------------
    # Assemble results in document order
    # -----------------------------------------------------------------------
    page_results: list[OCRPageResult] = []
    for i, gcv_result in enumerate(gcv_results):
        page_num = i + 1
        confidence = gcv_result.get("confidence")
        was_refined = i in pages_to_refine

        logger.info(
            "Page %d | conf=%s | refined=%s",
            page_num,
            f"{confidence:.4f}" if confidence is not None else "N/A",
            was_refined,
        )

        page_results.append(OCRPageResult(
            page_number=page_num,
            raw_text=gcv_result.get("raw_text", ""),
            normalized_text=gcv_result.get("raw_text", ""),
            refined_text=refined_texts.get(i, ""),
            perspective_corrected=False,
            confidence=confidence,
            word_confidences=gcv_result.get("word_confidences"),
            error=gcv_result.get("error"),
        ))

    total_elapsed = time.time() - t0
    logger.info("run_ocr complete: %d page(s) in %.2fs", n_pages, total_elapsed)

    metadata = {
        "filename": Path(file_path).name,
        "doc_id": doc_id,
        "total_pages": n_pages,
        "model_used": "google-cloud-vision",
        "gcv_feature": cfg["gcv_feature"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": round(total_elapsed, 2),
        "gcv_batch_seconds": round(gcv_elapsed, 2),
        "pdf_dpi": cfg["pdf_dpi"],
        "refine_enabled": cfg["refine_enabled"],
        "refine_confidence_threshold": cfg["refine_confidence_threshold"],
        "pages_refined": len(pages_to_refine),
        "pages_skipped_refine": len(skipped),
    }

    return OCRDocumentResult(metadata=metadata, pages=page_results)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_config(config: Optional[dict] = None) -> dict:
    from config.ocr import (  # noqa: PLC0415
        ALLOWED_EXTENSIONS,
        GCV_FEATURE,
        HIGH_CONFIDENCE_THRESHOLD,
        MAX_FILE_SIZE_MB,
        MAX_WORKERS,
        MEDIUM_CONFIDENCE_THRESHOLD,
        PDF_DPI,
        REFINEMENT_CONFIDENCE_THRESHOLD,
        REFINEMENT_ENABLED,
        REFINEMENT_TIMEOUT,
    )

    defaults: dict = {
        "gcv_feature": GCV_FEATURE,
        "pdf_dpi": PDF_DPI,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "allowed_extensions": ALLOWED_EXTENSIONS,
        "high_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "medium_threshold": MEDIUM_CONFIDENCE_THRESHOLD,
        "refine_enabled": REFINEMENT_ENABLED,
        "refine_timeout": REFINEMENT_TIMEOUT,
        "refine_confidence_threshold": REFINEMENT_CONFIDENCE_THRESHOLD,
        "max_workers": MAX_WORKERS,
    }

    if config:
        defaults.update(config)

    return defaults