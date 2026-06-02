# """
# DocumentProcessor.OCR.gcv_engine
# ----------------------------------
# Google Cloud Vision OCR engine (replaces QARI/Qwen2VL).

# Uses DOCUMENT_TEXT_DETECTION — the GCV feature optimised for dense,
# multi-column document text, which outperforms TEXT_DETECTION on Arabic
# legal documents with diacritics and right-to-left layout.

# Authentication
# --------------
# Set the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to the
# absolute path of your service-account JSON key file::

#     GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcv-sa.json

# Production upgrade path (GKE / Cloud Run):
#   Remove the env-var entirely and attach a Workload Identity service account
#   to the pod / revision.  The ``google-auth`` library picks it up
#   automatically — no code change required.

# Singleton
# ---------
# ``get_gcv_engine()`` returns a module-level singleton so the Vision client
# (and its gRPC channel) is created only once per process.
# """

# from __future__ import annotations

# import io
# import logging
# import threading
# from typing import List, Optional

# from PIL import Image

# from DocumentProcessor.OCR.models import WordConfidence
# from config.ocr import (
#     GCV_FEATURE,
#     HIGH_CONFIDENCE_THRESHOLD,
#     MEDIUM_CONFIDENCE_THRESHOLD,
# )

# logger = logging.getLogger(__name__)

# _engine_instance: Optional["GCVEngine"] = None
# _engine_lock = threading.Lock()


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _band(confidence: float) -> str:
#     """Map a raw confidence score to a UI colour band."""
#     if confidence >= HIGH_CONFIDENCE_THRESHOLD:
#         return "high"
#     if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
#         return "mid"
#     return "low"


# # ---------------------------------------------------------------------------
# # Error type
# # ---------------------------------------------------------------------------

# class GCVError(Exception):
#     """Raised when the GCV API returns an application-level error."""


# # ---------------------------------------------------------------------------
# # Engine
# # ---------------------------------------------------------------------------

# class GCVEngine:
#     """Thin, stateless wrapper around the Google Cloud Vision API client."""

#     def __init__(self, feature: str = GCV_FEATURE) -> None:
#         # Import deferred so the rest of the codebase can be imported without
#         # google-cloud-vision installed (e.g. during unit-test collection).
#         from google.cloud import vision  # noqa: PLC0415

#         self._vision = vision
#         self._feature_type = getattr(vision.Feature.Type, feature)
#         self._client = vision.ImageAnnotatorClient()
#         logger.info("GCVEngine initialised (feature=%s)", feature)

#     # ------------------------------------------------------------------
#     # Public
#     # ------------------------------------------------------------------

#     def ocr_page(
#         self,
#         pil_image: Image.Image,
#         page_number: int = 1,
#     ) -> dict:
#         """Run OCR on a single PIL Image.

#         Parameters
#         ----------
#         pil_image:
#             RGB PIL image for one document page.
#         page_number:
#             1-based index used for logging and stored inside each
#             ``WordConfidence`` object (needed when pages from multiple
#             files are merged downstream).

#         Returns
#         -------
#         dict with keys:
#             raw_text       -- full document text from GCV (str)
#             confidence     -- mean word confidence for the page (float | None)
#             word_confidences -- List[WordConfidence] | None
#             error          -- error message string | None
#         """
#         try:
#             image_bytes = self._pil_to_bytes(pil_image)
#             response = self._call_gcv(image_bytes)

#             if response.error.message:
#                 raise GCVError(
#                     f"GCV API error (code={response.error.code}): "
#                     f"{response.error.message}"
#                 )

#             annotation = response.full_text_annotation
#             raw_text: str = annotation.text if annotation else ""

#             word_confs = self._extract_word_confidences(annotation, page_number)

#             page_conf: Optional[float] = None
#             if word_confs:
#                 page_conf = round(
#                     sum(w.confidence for w in word_confs) / len(word_confs), 4
#                 )

#             logger.debug(
#                 "Page %d: %d words, mean_conf=%.4f",
#                 page_number,
#                 len(word_confs),
#                 page_conf or 0.0,
#             )

#             return {
#                 "raw_text": raw_text,
#                 "confidence": page_conf,
#                 "word_confidences": word_confs or None,
#                 "error": None,
#             }

#         except GCVError as exc:
#             logger.error("GCV API error on page %d: %s", page_number, exc)
#             return self._error_result(str(exc))

#         except Exception as exc:
#             logger.exception("GCV engine unexpected error on page %d: %s", page_number, exc)
#             return self._error_result(str(exc))

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _pil_to_bytes(image: Image.Image) -> bytes:
#         """Encode a PIL Image to PNG bytes without touching the filesystem."""
#         buf = io.BytesIO()
#         rgb = image.convert("RGB") if image.mode != "RGB" else image
#         rgb.save(buf, format="PNG")
#         return buf.getvalue()

#     def _call_gcv(self, image_bytes: bytes):
#         """Send image bytes to GCV and return the AnnotateImageResponse."""
#         image = self._vision.Image(content=image_bytes)
#         feature = self._vision.Feature(type_=self._feature_type)
#         request = self._vision.AnnotateImageRequest(
#             image=image, features=[feature]
#         )
#         return self._client.annotate_image(request)

#     def _extract_word_confidences(
#         self,
#         annotation,
#         page_number: int,
#     ) -> List[WordConfidence]:
#         """Walk the full-text annotation hierarchy and build WordConfidence list.

#         Hierarchy: FullTextAnnotation → pages → blocks → paragraphs → words
#         Each word carries its own confidence score from GCV.
#         """
#         words: List[WordConfidence] = []

#         if not annotation or not annotation.pages:
#             return words

#         for page in annotation.pages:
#             for block in page.blocks:
#                 for paragraph in block.paragraphs:
#                     for word in paragraph.words:
#                         word_text = "".join(
#                             symbol.text for symbol in word.symbols
#                         )
#                         if not word_text.strip():
#                             continue

#                         conf = round(float(word.confidence), 4)
#                         words.append(
#                             WordConfidence(
#                                 word=word_text,
#                                 confidence=conf,
#                                 band=_band(conf),
#                                 page_number=page_number,
#                             )
#                         )

#         return words

#     @staticmethod
#     def _error_result(message: str) -> dict:
#         return {
#             "raw_text": "",
#             "confidence": None,
#             "word_confidences": None,
#             "error": message,
#         }


# # ---------------------------------------------------------------------------
# # Singleton factory
# # ---------------------------------------------------------------------------

# def get_gcv_engine() -> GCVEngine:
#     """Return the process-level GCVEngine singleton (thread-safe)."""
#     global _engine_instance
#     if _engine_instance is None:
#         with _engine_lock:
#             if _engine_instance is None:
#                 _engine_instance = GCVEngine()
#     return _engine_instance


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

Performance notes
-----------------
``_pil_to_bytes`` encodes images as JPEG (quality 92) rather than PNG.

A 300 DPI A4 scan is ~2480 × 3508 px.  As **PNG** that is typically
8–18 MB, causing 10–25 s of upload latency even on a fast connection.
As **JPEG quality 92** the same image is 300 KB – 1.5 MB — a 10–15×
reduction.  GCV decodes the JPEG before running its vision models, so
OCR accuracy is unaffected.  The size guard (quality reduction → pixel
downscale) is now consolidated here; the pipeline no longer needs a
separate ``_enforce_size_limit`` call.
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

# GCV hard limit is 20 MB per image request.  We target 15 MB for headroom.
_GCV_MAX_BYTES = 15 * 1024 * 1024


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
            ``WordConfidence`` object.

        Returns
        -------
        dict with keys:
            raw_text         -- full document text from GCV (str)
            confidence       -- mean word confidence for the page (float | None)
            word_confidences -- List[WordConfidence] | None
            error            -- error message string | None
        """
        try:
            # _pil_to_bytes now handles JPEG encoding AND the size guard in
            # one pass — no separate _enforce_size_limit call needed.
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
            logger.exception(
                "GCV engine unexpected error on page %d: %s", page_number, exc
            )
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_bytes(
        image: Image.Image,
        max_bytes: int = _GCV_MAX_BYTES,
        jpeg_quality: int = 92,
    ) -> bytes:
        """Encode a PIL Image to JPEG bytes for the GCV upload.

        Why JPEG instead of PNG?
        ------------------------
        A 300 DPI A4 scan (≈2480 × 3508 px) encoded as **PNG** is typically
        8–18 MB.  The same image encoded as **JPEG at quality 92** is
        300 KB – 1.5 MB — a 10–15× reduction.  Because GCV decodes the
        JPEG internally before running inference, OCR accuracy is
        unaffected by the lossy compression.

        Size guard (replaces the old ``_enforce_size_limit`` in ocr_pipeline)
        -----------------------------------------------------------------------
        1. Try JPEG quality reduction: 92 → 85 → 75 → 60.
           Each step is fast (CPU only) and usually sufficient.
        2. If quality reduction alone doesn't fit the limit, iteratively
           downscale the pixel dimensions by 15 % until the payload fits.

        In practice step 2 only triggers for extremely dense 400 DPI colour
        scans.  Normal 300 DPI document scans pass after step 1 at quality 92.
        """
        rgb = image.convert("RGB") if image.mode != "RGB" else image

        # ---- Step 1: quality reduction (fast) --------------------------------
        for quality in (jpeg_quality, 85, 75, 60):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue()

        # ---- Step 2: pixel downscale (last resort) ----------------------------
        img = rgb
        while min(img.size) > 100:
            new_w = max(int(img.width * 0.85), 1)
            new_h = max(int(img.height * 0.85), 1)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            if buf.tell() <= max_bytes:
                logger.warning(
                    "Image downscaled to %dx%d (%.1f MB) to fit GCV upload limit",
                    new_w, new_h, buf.tell() / (1024 * 1024),
                )
                return buf.getvalue()

        return buf.getvalue()  # best-effort: upload whatever we produced

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
        """Walk the full-text annotation hierarchy and build WordConfidence list."""
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