"""
engine.py

Surya OCR engine wrapper with confidence scoring.

Uses Surya's two-stage pipeline:
1. Text Detection — finds text line bounding boxes with reading order
2. Text Recognition — recognizes characters with per-line confidence

Provides lazy model loading, GPU auto-detection, and batched inference
for efficient processing of multi-page documents.
"""

import logging
from typing import List, Optional

from PIL import Image

import config.ocr as config
from schemas import OCRLine, OCRPageResult, OCRWord

logger = logging.getLogger(__name__)


class SuryaOCREngine:
    """
    Wrapper around Surya's detection and recognition models.

    Models are loaded lazily on first use and cached for reuse.
    GPU is auto-detected with fallback to CPU.
    """

    def __init__(self):
        self._det_model = None
        self._rec_model = None
        self._det_processor = None
        self._rec_processor = None
        self._models_loaded = False

    def _load_models(self) -> None:
        """
        Lazy-load Surya detection and recognition models with compatibility patches.
        """
        if self._models_loaded:
            return

        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            import torch
        except ImportError:
            raise ImportError("surya-ocr is required. Install it with: pip install surya-ocr")

        logger.info("Loading Surya models with multi-stage compatibility patch...")

        # 1. Load Detection
        self._det_predictor = DetectionPredictor()

        # --- PATCH 1: Missing 'tasks' attribute ---
        if not hasattr(self._det_predictor, "tasks"):
            self._det_predictor.tasks = ["text_detection"]
            logger.info("Patch: Injected 'tasks' into DetectionPredictor")

        # --- PATCH 2: Missing 'bbox_size' in config ---
        if not hasattr(self._det_predictor.model.config, "bbox_size"):
            # 640 is the standard internal resolution for Surya
            self._det_predictor.model.config.bbox_size = 640
            logger.info("Patch: Injected 'bbox_size' into Detection config")

        # 2. Load Recognition using the patched detector
        self._rec_predictor = RecognitionPredictor(self._det_predictor)

        self._models_loaded = True
        logger.info("Surya models loaded successfully")

    def process(self, images: List[Image.Image]) -> List[OCRPageResult]:
        """
        Run the full Surya pipeline (detect + recognize) and map to project schemas.
        """
        self._load_models()

        # 2. Stage 1: Text Detection
        # Returns a list of DetectionResult objects
        det_results = self._det_predictor(images)

        # 3. Stage 2: Text Recognition
        # Extract raw coordinates [x1, y1, x2, y2] from Box objects
        all_bboxes = [[box.bbox for box in dr.bboxes] for dr in det_results]
        
        # FIX: The keyword is 'langs', not 'languages'
        rec_results = self._rec_predictor(
            images,  
            bboxes=all_bboxes
        )

        page_results = []
        for i, rec_result in enumerate(rec_results):
            page_num = i + 1
            lines = []

            # 4. Map Surya's RecognitionResult to OCRLine/OCRWord schema
            for text_line in rec_result.text_lines:
                text = text_line.text.strip()
                if not text:
                    continue

                bbox = text_line.bbox
                corner_points = [
                    (float(bbox[0]), float(bbox[1])), # TL
                    (float(bbox[2]), float(bbox[1])), # TR
                    (float(bbox[2]), float(bbox[3])), # BR
                    (float(bbox[0]), float(bbox[3])), # BL
                ]

                # Create schema objects defined in your project
                word = OCRWord(
                    text=text,
                    bbox=corner_points,
                    confidence=float(getattr(text_line, "confidence", 0.0))
                )

                line = OCRLine(
                    words=[word],
                    text=text,
                    confidence=float(getattr(text_line, "confidence", 0.0))
                )
                lines.append(line)

            # Compute stats using the helper function at the bottom of your file
            raw_text = "\n".join(l.text for l in lines)
            page_conf = _compute_page_confidence(lines)

            # Construct the final result your pipeline expects
            page_result = OCRPageResult(
                page_number=page_num,  # Ensure this matches your schemas.py
                lines=lines,
                raw_text=raw_text,
                confidence=page_conf,
                warnings=[],
                has_errors=False
            )
            page_results.append(page_result)

        return page_results

    def _compute_page_confidence_internal(self, lines: List[OCRLine]) -> float:
        """Helper to compute confidence if the global one isn't reachable."""
        if not lines: return 0.0
        total_chars = sum(len(line.text) for line in lines)
        if total_chars == 0: return 0.0
        weighted_sum = sum(line.confidence * len(line.text) for line in lines)
        return weighted_sum / total_chars

    def _process_batch(
        self,
        images: List[Image.Image],
        batch_text_detection,
        batch_recognition,
        page_offset: int = 0,
    ) -> List[OCRPageResult]:
        """Process a single batch of images through detection and recognition."""
        page_results = []

        for i, image in enumerate(images):
            page_num = page_offset + i + 1

            try:
                page_result = self._process_single_image(
                    image, batch_text_detection, batch_recognition, page_num
                )
            except Exception as e:
                logger.error("Surya engine error on page %d: %s", page_num, e)
                page_result = OCRPageResult(
                    page_number=page_num,
                    lines=[],
                    raw_text="",
                    confidence=0.0,
                    warnings=[f"OCR engine error: {str(e)}"],
                    has_errors=True,
                )

            page_results.append(page_result)

        return page_results

    def _process_single_image(
        self,
        image: Image.Image,
        batch_text_detection,
        batch_recognition,
        page_number: int,
    ) -> OCRPageResult:
        """Process a single image through Surya detection and recognition."""
        warnings: List[str] = []

        # Stage 1: Text Detection
        det_results = batch_text_detection(
            [image], self._det_model, self._det_processor
        )

        if not det_results or not det_results[0].bboxes:
            return OCRPageResult(
                page_number=page_number,
                lines=[],
                raw_text="",
                confidence=0.0,
                warnings=["No text detected on page"],
                has_errors=False,
            )

        # Stage 2: Text Recognition
        langs = [[config.OCR_LANGUAGE]] * 1  # One language list per image
        rec_results = batch_recognition(
            [image], langs, self._rec_model, self._rec_processor, det_results
        )

        if not rec_results or not rec_results[0].text_lines:
            return OCRPageResult(
                page_number=page_number,
                lines=[],
                raw_text="",
                confidence=0.0,
                warnings=["Recognition produced no text"],
                has_errors=False,
            )

        # Convert Surya results to our schema
        rec_result = rec_results[0]
        lines = []

        for text_line in rec_result.text_lines:
            text = text_line.text.strip()
            if not text:
                continue

            confidence = getattr(text_line, "confidence", 0.0)
            bbox = getattr(text_line, "bbox", [0, 0, 0, 0])

            # Convert bbox [x1, y1, x2, y2] to corner points format
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                corner_points = [
                    (float(x1), float(y1)),
                    (float(x2), float(y1)),
                    (float(x2), float(y2)),
                    (float(x1), float(y2)),
                ]
            else:
                corner_points = [(0.0, 0.0)] * 4

            # Create a single word per line (Surya returns line-level text)
            word = OCRWord(
                text=text,
                bbox=corner_points,
                confidence=float(confidence),
            )

            line = OCRLine(
                words=[word],
                text=text,
                confidence=float(confidence),
            )

            # Flag low-confidence lines
            if confidence < config.MEDIUM_CONFIDENCE_THRESHOLD:
                warnings.append(
                    f"Low confidence ({confidence:.2f}) for: '{text[:50]}'"
                )

            lines.append(line)

        # Compute page-level confidence
        page_confidence = _compute_page_confidence(lines)

        # Build raw text
        raw_text = "\n".join(line.text for line in lines)

        return OCRPageResult(
            page_number=page_number,
            lines=lines,
            raw_text=raw_text,
            confidence=page_confidence,
            warnings=warnings,
            has_errors=False,
        )

    def reset(self) -> None:
        """Release models and free memory."""
        self._det_model = None
        self._rec_model = None
        self._det_processor = None
        self._rec_processor = None
        self._models_loaded = False
        logger.info("Surya models released")


# Module-level singleton engine
_engine: Optional[SuryaOCREngine] = None


def get_engine() -> SuryaOCREngine:
    """Get or create the singleton Surya OCR engine."""
    global _engine
    if _engine is None:
        _engine = SuryaOCREngine()
    return _engine


def reset_engine() -> None:
    """Reset the singleton engine (useful for testing)."""
    global _engine
    if _engine is not None:
        _engine.reset()
    _engine = None


def run_ocr(images: List[Image.Image]) -> List[OCRPageResult]:
    """
    Run Surya OCR on a list of PIL Images.

    This is the main entry point for the engine module.
    Uses the singleton engine with lazy model loading.

    Args:
        images: List of PIL Images in RGB mode.

    Returns:
        List of OCRPageResult, one per input image.
    """
    engine = get_engine()
    return engine.process(images)


def _compute_page_confidence(lines: List[OCRLine]) -> float:
    """
    Compute page-level confidence as a weighted average of line confidences.
    Weight is proportional to the number of characters in each line.
    """
    if not lines:
        return 0.0

    total_chars = sum(len(line.text) for line in lines)
    if total_chars == 0:
        return 0.0

    weighted_sum = sum(line.confidence * len(line.text) for line in lines)
    return weighted_sum / total_chars
