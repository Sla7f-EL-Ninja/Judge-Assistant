"""
DocumentProcessor.OCR.gcv_engine
----------------------------------
Google Cloud Vision OCR engine.

Performance notes — v4
-----------------------
``ocr_batch()`` now chunks large documents to stay within GCV's hard limit
of 16 images per ``batch_annotate_images`` request.

Previously, sending 20 pages in one call raised:
    400 Too many images per request

Fix: the method splits ``pil_images`` into chunks of ``GCV_BATCH_LIMIT``
(default 16), fires one batch API call per chunk sequentially, and
concatenates the results in page order.

Why sequential chunks rather than concurrent chunk calls?
  Sending multiple batch calls concurrently would re-introduce the
  HTTP/2 head-of-line blocking that caused the ~100 s regressions in
  the per-page concurrent approach.  Sequential chunks keep one
  in-flight request at a time, which is what GCV processes most
  efficiently.

Timing estimate for a 20-page document (200 DPI):
  Chunk 1 (pages 1-16):  ~25-35 s
  Chunk 2 (pages 17-20): ~15 s
  Total GCV:             ~40-50 s  →  ~2-2.5 s / page
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import time
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

_GCV_MAX_BYTES = 15 * 1024 * 1024      # 15 MB (hard limit is 20 MB)
GCV_BATCH_LIMIT = 16                   # GCV's hard cap on images per batch call


def _band(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "mid"
    return "low"


class GCVError(Exception):
    pass


class GCVEngine:
    """Thin, stateless wrapper around the Google Cloud Vision API client."""

    def __init__(self, feature: str = GCV_FEATURE) -> None:
        from google.cloud import vision  # noqa: PLC0415
        self._vision = vision
        self._feature_type = getattr(vision.Feature.Type, feature)
        self._client = vision.ImageAnnotatorClient()
        logger.info("GCVEngine initialised (feature=%s)", feature)

    # ------------------------------------------------------------------
    # Batch API  (primary path for multi-page documents)
    # ------------------------------------------------------------------

    def ocr_batch(
        self,
        pil_images: List[Image.Image],
        max_encode_workers: int = 4,
        chunk_size: int = GCV_BATCH_LIMIT,
    ) -> List[dict]:
        """Run OCR on all pages using chunked batch API calls.

        GCV's ``batch_annotate_images`` accepts at most ``GCV_BATCH_LIMIT``
        (16) images per request.  This method:

        1. Encodes all images to JPEG in parallel (libjpeg releases the GIL).
        2. Splits the encoded bytes into chunks of at most ``chunk_size``.
        3. Fires one ``batch_annotate_images`` call per chunk, sequentially.
        4. Returns all results concatenated in the original page order.

        Parameters
        ----------
        pil_images:
            Ordered list of RGB PIL images, one per document page.
        max_encode_workers:
            Thread-pool size for parallel JPEG encoding.
        chunk_size:
            Images per GCV batch call.  Capped internally at
            ``GCV_BATCH_LIMIT`` (16) regardless of what is passed.

        Returns
        -------
        list[dict]
            One result dict per input image, in input order.
            Keys: ``raw_text``, ``confidence``, ``word_confidences``, ``error``.
        """
        if not pil_images:
            return []

        n = len(pil_images)
        chunk_size = min(chunk_size, GCV_BATCH_LIMIT)   # never exceed GCV cap

        # ---- Step 1: encode all images in parallel -----------------------
        t_enc = time.time()
        workers = min(n, max_encode_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as enc_pool:
            image_bytes_list: List[bytes] = list(
                enc_pool.map(self._pil_to_bytes, pil_images)
            )
        logger.info(
            "JPEG encoding: %d page(s) in %.2fs (workers=%d)",
            n, time.time() - t_enc, workers,
        )

        # ---- Step 2: chunk + call GCV ------------------------------------
        n_chunks = (n + chunk_size - 1) // chunk_size
        all_results: List[dict] = []

        t_total_api = time.time()
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end   = min(start + chunk_size, n)
            chunk_bytes = image_bytes_list[start:end]
            chunk_len   = len(chunk_bytes)

            logger.info(
                "GCV batch chunk %d/%d: pages %d–%d (%d image(s)) …",
                chunk_idx + 1, n_chunks, start + 1, end, chunk_len,
            )

            requests = [
                self._vision.AnnotateImageRequest(
                    image=self._vision.Image(content=b),
                    features=[self._vision.Feature(type_=self._feature_type)],
                )
                for b in chunk_bytes
            ]

            t_chunk = time.time()
            try:
                batch_resp = self._client.batch_annotate_images(requests=requests)
            except Exception as exc:
                logger.error(
                    "GCV batch chunk %d/%d failed: %s", chunk_idx + 1, n_chunks, exc
                )
                all_results.extend(
                    [self._error_result(str(exc))] * chunk_len
                )
                continue

            logger.info(
                "GCV batch chunk %d/%d returned in %.2fs",
                chunk_idx + 1, n_chunks, time.time() - t_chunk,
            )

            # ---- Step 3: unpack results for this chunk -------------------
            for i, response in enumerate(batch_resp.responses):
                page_num = start + i + 1
                if response.error.message:
                    logger.error(
                        "GCV error on page %d (code=%d): %s",
                        page_num, response.error.code, response.error.message,
                    )
                    all_results.append(self._error_result(response.error.message))
                    continue

                annotation = response.full_text_annotation
                raw_text: str = annotation.text if annotation else ""
                word_confs = self._extract_word_confidences(annotation, page_num)

                page_conf: Optional[float] = None
                if word_confs:
                    page_conf = round(
                        sum(w.confidence for w in word_confs) / len(word_confs), 4
                    )

                logger.debug(
                    "Page %d: %d words, conf=%.4f",
                    page_num, len(word_confs), page_conf or 0.0,
                )
                all_results.append({
                    "raw_text": raw_text,
                    "confidence": page_conf,
                    "word_confidences": word_confs or None,
                    "error": None,
                })

        total_api_s = time.time() - t_total_api
        logger.info(
            "GCV all chunks done: %d page(s) in %.2fs across %d chunk(s) (%.1fs/page)",
            n, total_api_s, n_chunks, total_api_s / max(n, 1),
        )
        return all_results

    # ------------------------------------------------------------------
    # Single-page API  (kept for backward compat / single-image use)
    # ------------------------------------------------------------------

    def ocr_page(self, pil_image: Image.Image, page_number: int = 1) -> dict:
        """Run OCR on a single PIL Image (one API call)."""
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
            logger.exception("GCV engine error on page %d: %s", page_number, exc)
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

        Quality reduction (92→85→75→60) then pixel downscaling as a
        last resort to stay within GCV's 20 MB per-image limit.
        """
        rgb = image.convert("RGB") if image.mode != "RGB" else image

        for quality in (jpeg_quality, 85, 75, 60):
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= max_bytes:
                return buf.getvalue()

        img = rgb
        while min(img.size) > 100:
            new_w = max(int(img.width * 0.85), 1)
            new_h = max(int(img.height * 0.85), 1)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75, optimize=True)
            if buf.tell() <= max_bytes:
                logger.warning(
                    "Image downscaled to %dx%d (%.1f MB) for GCV",
                    new_w, new_h, buf.tell() / (1024 * 1024),
                )
                return buf.getvalue()

        return buf.getvalue()

    def _call_gcv(self, image_bytes: bytes):
        image = self._vision.Image(content=image_bytes)
        feature = self._vision.Feature(type_=self._feature_type)
        return self._client.annotate_image(
            self._vision.AnnotateImageRequest(image=image, features=[feature])
        )

    def _extract_word_confidences(
        self, annotation, page_number: int
    ) -> List[WordConfidence]:
        words: List[WordConfidence] = []
        if not annotation or not annotation.pages:
            return words
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join(s.text for s in word.symbols)
                        if not word_text.strip():
                            continue
                        conf = round(float(word.confidence), 4)
                        words.append(WordConfidence(
                            word=word_text,
                            confidence=conf,
                            band=_band(conf),
                            page_number=page_number,
                        ))
        return words

    @staticmethod
    def _error_result(message: str) -> dict:
        return {
            "raw_text": "",
            "confidence": None,
            "word_confidences": None,
            "error": message,
        }


def get_gcv_engine() -> GCVEngine:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = GCVEngine()
    return _engine_instance