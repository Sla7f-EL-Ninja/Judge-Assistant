"""
ocr_pipeline.py

Main orchestrator for the OCR module.

Coordinates the full pipeline: loading -> preprocessing -> Surya OCR -> post-processing.
Supports single-file and batch processing, with an option to return results
formatted for the Summarization pipeline's Node 0.

Uses Surya's native batch processing for efficient GPU inference.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from PIL import Image

import config.ocr as config
from engine import run_ocr
from postprocessor import postprocess_document_pages, postprocess_page
from preprocessor import preprocess_image
from schemas import OCRDocumentResult, OCRPageResult
from utils import load_images

logger = logging.getLogger(__name__)


def process_document(
    file_path: Union[str, List[str]],
    return_for_node0: bool = False,
    doc_id: Optional[str] = None,
) -> Union[OCRDocumentResult, List[Dict]]:
    """
    Process a document through the full OCR pipeline.

    Args:
        file_path: Path to an image or PDF file, or list of image paths
            for a multi-page document.
        return_for_node0: If True, return in the format expected by
            SummarizationState.documents: [{"raw_text": "...", "doc_id": "..."}]
        doc_id: Optional document identifier. Auto-generated if not provided.

    Returns:
        OCRDocumentResult with structured results, or a list of dicts
        for Node 0 integration if return_for_node0 is True.
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())[:8]

    logger.info("Processing document: %s (doc_id=%s)", file_path, doc_id)

    # 1. Load images
    if isinstance(file_path, list):
        all_images: List[Image.Image] = []
        for fp in file_path:
            all_images.extend(load_images(fp))
        display_path = str(file_path[0]) if file_path else "unknown"
    else:
        all_images = load_images(file_path)
        display_path = str(file_path)

    logger.info("Loaded %d page(s)", len(all_images))

    # 2. Preprocess each image
    preprocessed_images: List[Image.Image] = []
    for i, image in enumerate(all_images):
        logger.info("Preprocessing page %d/%d", i + 1, len(all_images))
        preprocessed = preprocess_image(image)
        preprocessed_images.append(preprocessed)

    # 3. Run Surya OCR (batched)
    logger.info("Running Surya OCR on %d page(s)...", len(preprocessed_images))
    page_results = run_ocr(preprocessed_images)

    # 4. Post-process each page
    all_warnings: List[str] = []
    postprocessed_pages: List[OCRPageResult] = []

    for page_result in page_results:
        corrected = postprocess_page(page_result)
        postprocessed_pages.append(corrected)
        all_warnings.extend(corrected.warnings)

    # 5. Document-level post-processing (header/footer removal)
    postprocessed_pages = postprocess_document_pages(postprocessed_pages)

    # 6. Combine results
    combined_text = "\n\n".join(
        page.raw_text for page in postprocessed_pages if page.raw_text
    )

    overall_confidence = _compute_document_confidence(postprocessed_pages)

    result = OCRDocumentResult(
        file_path=display_path,
        doc_id=doc_id,
        pages=postprocessed_pages,
        raw_text=combined_text,
        total_pages=len(postprocessed_pages),
        overall_confidence=overall_confidence,
        warnings=all_warnings,
    )

    logger.info(
        "Document processed: %d pages, confidence=%.2f, warnings=%d",
        result.total_pages,
        result.overall_confidence,
        len(result.warnings),
    )

    if return_for_node0:
        return [{"raw_text": result.raw_text, "doc_id": result.doc_id}]

    return result


def process_batch(
    file_paths: List[str],
    return_for_node0: bool = False,
    max_workers: Optional[int] = None,
) -> Union[List[OCRDocumentResult], List[Dict]]:
    """
    Process multiple documents.

    Note: Surya uses GPU internally with batched inference, so documents
    are processed sequentially to avoid GPU memory contention. The batch
    size for Surya's internal batching is controlled by config.SURYA_BATCH_SIZE.

    Args:
        file_paths: List of file paths to process.
        return_for_node0: If True, return in Node 0 format.
        max_workers: Number of concurrent workers (used for CPU-bound
            preprocessing only, not for GPU inference).

    Returns:
        List of OCRDocumentResult objects, or list of dicts for Node 0.
    """
    results: List[OCRDocumentResult] = []

    for fp in file_paths:
        try:
            result = process_document(fp, return_for_node0=False)
            results.append(result)
        except Exception as e:
            logger.error("Failed to process %s: %s", fp, e)
            results.append(
                OCRDocumentResult(
                    file_path=str(fp),
                    doc_id=str(uuid.uuid4())[:8],
                    pages=[],
                    raw_text="",
                    total_pages=0,
                    overall_confidence=0.0,
                    warnings=[f"Processing failed: {str(e)}"],
                )
            )

    if return_for_node0:
        return [
            {"raw_text": r.raw_text, "doc_id": r.doc_id}
            for r in results
        ]

    return results


def _compute_document_confidence(pages: List[OCRPageResult]) -> float:
    """
    Compute document-level confidence as a weighted average of page confidences.
    Weight is proportional to the text length on each page.
    """
    if not pages:
        return 0.0

    total_chars = sum(len(p.raw_text) for p in pages)
    if total_chars == 0:
        return 0.0

    weighted_sum = sum(p.confidence * len(p.raw_text) for p in pages)
    return weighted_sum / total_chars
