"""
DocumentProcessor.OCR.gcv_engine
----------------------------------
Google Cloud Vision OCR engine (replaces QARI/Qwen2VL).

Uses DOCUMENT_TEXT_DETECTION — the GCV feature optimised for dense,
multi-column document text, which outperforms TEXT_DETECTION on Arabic
legal documents with diacritics and right-to-left layout.

Authentication
--------------
Set the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to the
absolute path of your service-account JSON key file::

    GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcv-sa.json

Production upgrade path (GKE / Cloud Run):
  Remove the env-var entirely and attach a Workload Identity service account
  to the pod / revision.  The ``google-auth`` library picks it up
  automatically — no code change required.

Singleton
---------
``get_gcv_engine()`` returns a module-level singleton so the Vision client
(and its gRPC channel) is created only once per process.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import List, Optional

from PIL import Image

from DocumentProcessor.OCR.models import WordConfidence
from config.ocr import (
    GCV_FEATURE,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_engine_instance: Optional["GCVEngine"] = None
_engine_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band(confidence: float) -> str:
    """Map a raw confidence score to a UI colour band."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "mid"
    return "low"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class GCVError(Exception):
    """Raised when the GCV API returns an application-level error."""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GCVEngine:
    """Thin, stateless wrapper around the Google Cloud Vision API client."""

    def __init__(self, feature: str = GCV_FEATURE) -> None:
        # Import deferred so the rest of the codebase can be imported without
        # google-cloud-vision installed (e.g. during unit-test collection).
        from google.cloud import vision  # noqa: PLC0415

        self._vision = vision
        self._feature_type = getattr(vision.Feature.Type, feature)
        self._client = vision.ImageAnnotatorClient()
        logger.info("GCVEngine initialised (feature=%s)", feature)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ocr_page(
        self,
        pil_image: Image.Image,
        page_number: int = 1,
    ) -> dict:
        """Run OCR on a single PIL Image.

        Parameters
        ----------
        pil_image:
            RGB PIL image for one document page.
        page_number:
            1-based index used for logging and stored inside each
            ``WordConfidence`` object (needed when pages from multiple
            files are merged downstream).

        Returns
        -------
        dict with keys:
            raw_text       -- full document text from GCV (str)
            confidence     -- mean word confidence for the page (float | None)
            word_confidences -- List[WordConfidence] | None
            error          -- error message string | None
        """
        try:
            image_bytes = self._pil_to_bytes(pil_image)
            response = self._call_gcv(image_bytes)

            if response.error.message:
                raise GCVError(
                    f"GCV API error (code={response.error.code}): "
                    f"{response.error.message}"
                )

            annotation = response.full_text_annotation
            raw_text: str = annotation.text if annotation else ""

            word_confs = self._extract_word_confidences(annotation, page_number)

            page_conf: Optional[float] = None
            if word_confs:
                page_conf = round(
                    sum(w.confidence for w in word_confs) / len(word_confs), 4
                )

            logger.debug(
                "Page %d: %d words, mean_conf=%.4f",
                page_number,
                len(word_confs),
                page_conf or 0.0,
            )

            return {
                "raw_text": raw_text,
                "confidence": page_conf,
                "word_confidences": word_confs or None,
                "error": None,
            }

        except GCVError as exc:
            logger.error("GCV API error on page %d: %s", page_number, exc)
            return self._error_result(str(exc))

        except Exception as exc:
            logger.exception("GCV engine unexpected error on page %d: %s", page_number, exc)
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_bytes(image: Image.Image) -> bytes:
        """Encode a PIL Image to PNG bytes without touching the filesystem."""
        buf = io.BytesIO()
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        rgb.save(buf, format="PNG")
        return buf.getvalue()

    def _call_gcv(self, image_bytes: bytes):
        """Send image bytes to GCV and return the AnnotateImageResponse."""
        image = self._vision.Image(content=image_bytes)
        feature = self._vision.Feature(type_=self._feature_type)
        request = self._vision.AnnotateImageRequest(
            image=image, features=[feature]
        )
        return self._client.annotate_image(request)

    def _extract_word_confidences(
        self,
        annotation,
        page_number: int,
    ) -> List[WordConfidence]:
        """Walk the full-text annotation hierarchy and build WordConfidence list.

        Hierarchy: FullTextAnnotation → pages → blocks → paragraphs → words
        Each word carries its own confidence score from GCV.
        """
        words: List[WordConfidence] = []

        if not annotation or not annotation.pages:
            return words

        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join(
                            symbol.text for symbol in word.symbols
                        )
                        if not word_text.strip():
                            continue

                        conf = round(float(word.confidence), 4)
                        words.append(
                            WordConfidence(
                                word=word_text,
                                confidence=conf,
                                band=_band(conf),
                                page_number=page_number,
                            )
                        )

        return words

    @staticmethod
    def _error_result(message: str) -> dict:
        return {
            "raw_text": "",
            "confidence": None,
            "word_confidences": None,
            "error": message,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_gcv_engine() -> GCVEngine:
    """Return the process-level GCVEngine singleton (thread-safe)."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = GCVEngine()
    return _engine_instance
