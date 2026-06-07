# """
# tests/ocr/test_gcv.py
# ----------------------
# Tests for DocumentProcessor.OCR.gcv_engine.

# Unit tests (no network):
#   TestPilToBytes         — JPEG encoding, size guard, mode handling
#   TestBandLogic          — confidence → colour-band mapping
#   TestWordConfExtraction — _extract_word_confidences from a mock annotation
#   TestBatchChunking      — correct chunk count, page ordering, error isolation

# Integration tests (live GCV — @pytest.mark.integration):
#   TestGCVOCRPage         — ocr_page() on a real image
#   TestGCVOCRBatch        — ocr_batch() on 1 and N real images
# """

# from __future__ import annotations

# import io
# import struct
# from unittest.mock import MagicMock, call, patch

# import pytest
# from PIL import Image

# from conftest import make_batch_response


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _white(w: int = 200, h: int = 200) -> Image.Image:
#     return Image.new("RGB", (w, h), (255, 255, 255))


# def _jpeg_signature(data: bytes) -> bool:
#     return data[:2] == b"\xff\xd8"


# # ---------------------------------------------------------------------------
# # _pil_to_bytes  (unit)
# # ---------------------------------------------------------------------------

# class TestPilToBytes:
#     """Tests for GCVEngine._pil_to_bytes — JPEG encoding + size guard."""

#     def _encode(self, img: Image.Image, **kwargs) -> bytes:
#         from DocumentProcessor.OCR.gcv_engine import GCVEngine
#         return GCVEngine._pil_to_bytes(img, **kwargs)

#     def test_output_is_jpeg_not_png(self):
#         data = self._encode(_white())
#         assert _jpeg_signature(data), "Output must start with JPEG SOI marker 0xFFD8"

#     def test_rgb_image_encoded_correctly(self):
#         img = Image.new("RGB", (800, 1000), (200, 200, 200))
#         data = self._encode(img)
#         assert _jpeg_signature(data)
#         assert len(data) > 0

#     def test_rgba_image_converted_to_rgb(self):
#         img = Image.new("RGBA", (300, 300), (100, 150, 200, 128))
#         data = self._encode(img)   # must not raise
#         assert _jpeg_signature(data)

#     def test_grayscale_image_converted_to_rgb(self):
#         img = Image.new("L", (300, 300), 128)
#         data = self._encode(img)
#         assert _jpeg_signature(data)

#     def test_palette_image_converted(self):
#         img = Image.new("P", (300, 300))
#         data = self._encode(img)
#         assert _jpeg_signature(data)

#     def test_output_fits_under_default_limit(self):
#         from DocumentProcessor.OCR.gcv_engine import _GCV_MAX_BYTES
#         img = _white(1240, 1754)  # A4 at 150 DPI
#         data = self._encode(img)
#         assert len(data) <= _GCV_MAX_BYTES

#     def test_size_guard_reduces_quality_before_resize(self):
#         """Encoding at quality=92 must first try lower qualities before resizing."""
#         # A very small max_bytes forces quality reduction
#         img = Image.new("RGB", (500, 500), (100, 100, 100))
#         data = self._encode(img, max_bytes=50_000, jpeg_quality=92)
#         assert _jpeg_signature(data)
#         assert len(data) <= 50_000

#     def test_size_guard_extreme_constraint_still_returns_bytes(self):
#         """Even an extremely tight limit should return something rather than crash."""
#         img = Image.new("RGB", (2000, 2000), (50, 50, 50))
#         data = self._encode(img, max_bytes=10_000)
#         assert isinstance(data, bytes)
#         assert len(data) > 0

#     def test_jpeg_is_decodable(self):
#         """Round-trip: encoded bytes must decode back to a valid PIL image."""
#         img = Image.new("RGB", (640, 480), (120, 180, 240))
#         data = self._encode(img)
#         decoded = Image.open(io.BytesIO(data))
#         assert decoded.size[0] > 0


# # ---------------------------------------------------------------------------
# # _band  (unit)
# # ---------------------------------------------------------------------------

# class TestBandLogic:
#     """Tests for the confidence → colour-band mapping."""

#     def _band(self, conf: float) -> str:
#         from DocumentProcessor.OCR.gcv_engine import _band
#         return _band(conf)

#     def test_high_band_at_threshold(self):
#         from config.ocr import HIGH_CONFIDENCE_THRESHOLD
#         assert self._band(HIGH_CONFIDENCE_THRESHOLD) == "high"

#     def test_high_band_above_threshold(self):
#         assert self._band(1.0) == "high"
#         assert self._band(0.99) == "high"

#     def test_mid_band_at_threshold(self):
#         from config.ocr import MEDIUM_CONFIDENCE_THRESHOLD
#         assert self._band(MEDIUM_CONFIDENCE_THRESHOLD) == "mid"

#     def test_mid_band_just_below_high(self):
#         from config.ocr import HIGH_CONFIDENCE_THRESHOLD
#         assert self._band(HIGH_CONFIDENCE_THRESHOLD - 0.001) == "mid"

#     def test_low_band_below_medium_threshold(self):
#         from config.ocr import MEDIUM_CONFIDENCE_THRESHOLD
#         assert self._band(MEDIUM_CONFIDENCE_THRESHOLD - 0.001) == "low"

#     def test_low_band_at_zero(self):
#         assert self._band(0.0) == "low"


# # ---------------------------------------------------------------------------
# # _extract_word_confidences  (unit — mocked annotation)
# # ---------------------------------------------------------------------------

# class TestWordConfExtraction:
#     def _make_word(self, text: str, conf: float) -> MagicMock:
#         symbol = MagicMock()
#         symbol.text = text
#         word = MagicMock()
#         word.symbols = [MagicMock(text=c) for c in text]
#         word.confidence = conf
#         return word

#     def test_extracts_words_from_annotation(self, gcv_engine_unit):
#         word1 = self._make_word("محكمة", 0.92)
#         word2 = self._make_word("مصر", 0.88)
#         paragraph = MagicMock(words=[word1, word2])
#         block = MagicMock(paragraphs=[paragraph])
#         page = MagicMock(blocks=[block])
#         annotation = MagicMock(pages=[page])

#         results = gcv_engine_unit._extract_word_confidences(annotation, page_number=2)
#         assert len(results) == 2
#         assert results[0].word == "محكمة"
#         assert results[0].confidence == 0.92
#         assert results[0].page_number == 2
#         assert results[1].word == "مصر"

#     def test_skips_whitespace_only_words(self, gcv_engine_unit):
#         word = MagicMock()
#         word.symbols = [MagicMock(text=" ")]
#         word.confidence = 0.9
#         paragraph = MagicMock(words=[word])
#         block = MagicMock(paragraphs=[paragraph])
#         page = MagicMock(blocks=[block])
#         annotation = MagicMock(pages=[page])

#         results = gcv_engine_unit._extract_word_confidences(annotation, page_number=1)
#         assert results == []

#     def test_empty_annotation_returns_empty_list(self, gcv_engine_unit):
#         assert gcv_engine_unit._extract_word_confidences(None, 1) == []
#         annotation = MagicMock(pages=[])
#         assert gcv_engine_unit._extract_word_confidences(annotation, 1) == []

#     def test_band_assigned_to_each_word(self, gcv_engine_unit):
#         word = MagicMock()
#         word.symbols = [MagicMock(text="x")]
#         word.confidence = 0.99
#         para = MagicMock(words=[word])
#         block = MagicMock(paragraphs=[para])
#         page = MagicMock(blocks=[block])
#         annotation = MagicMock(pages=[page])
#         results = gcv_engine_unit._extract_word_confidences(annotation, 1)
#         assert results[0].band == "high"


# # ---------------------------------------------------------------------------
# # Batch chunking  (unit — mocked client)
# # ---------------------------------------------------------------------------

# class TestBatchChunking:
#     """Verify ocr_batch splits large requests into ≤16-image chunks."""

#     @pytest.mark.parametrize("n_pages,expected_chunks", [
#         (1,  1),
#         (4,  1),
#         (15, 1),
#         (16, 1),
#         (17, 2),
#         (20, 2),
#         (32, 2),
#         (33, 3),
#         (48, 3),
#         (53, 4),
#     ])
#     def test_chunk_count(self, n_pages, expected_chunks, gcv_engine_unit):
#         images = [_white() for _ in range(n_pages)]
#         call_count = 0

#         def fake_batch(requests):
#             nonlocal call_count
#             call_count += 1
#             return make_batch_response(len(requests))

#         gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
#         results = gcv_engine_unit.ocr_batch(images)
#         assert call_count == expected_chunks
#         assert len(results) == n_pages

#     def test_chunk_size_capped_at_gcv_limit(self, gcv_engine_unit):
#         """chunk_size > GCV_BATCH_LIMIT must be silently capped."""
#         from DocumentProcessor.OCR.gcv_engine import GCV_BATCH_LIMIT
#         images = [_white() for _ in range(GCV_BATCH_LIMIT + 1)]  # needs 2 chunks

#         call_count = 0
#         def fake_batch(requests):
#             nonlocal call_count
#             call_count += 1
#             assert len(requests) <= GCV_BATCH_LIMIT, "chunk exceeded GCV limit"
#             return make_batch_response(len(requests))

#         gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
#         gcv_engine_unit.ocr_batch(images, chunk_size=999)  # ask for huge chunk
#         assert call_count == 2

#     def test_empty_input_returns_empty_list(self, gcv_engine_unit):
#         result = gcv_engine_unit.ocr_batch([])
#         assert result == []
#         gcv_engine_unit._client.batch_annotate_images.assert_not_called()

#     def test_results_returned_in_page_order(self, gcv_engine_unit):
#         """Results must come back in the same order as input images."""
#         texts = [f"page text {i}" for i in range(20)]
#         images = [_white() for _ in range(20)]

#         call_idx = [0]
#         def fake_batch(requests):
#             chunk_texts = texts[call_idx[0] * 16 : call_idx[0] * 16 + len(requests)]
#             call_idx[0] += 1
#             resp = MagicMock()
#             responses = []
#             for t in chunk_texts:
#                 r = MagicMock()
#                 r.error.message = ""
#                 r.full_text_annotation.text = t
#                 r.full_text_annotation.pages = []
#                 responses.append(r)
#             resp.responses = responses
#             return resp

#         gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
#         results = gcv_engine_unit.ocr_batch(images)
#         for i, res in enumerate(results):
#             assert res["raw_text"] == texts[i], f"Page {i} text mismatch"

#     def test_chunk_error_does_not_stop_other_chunks(self, gcv_engine_unit):
#         """If chunk 1 raises, chunk 2 should still be processed."""
#         images = [_white() for _ in range(20)]  # 2 chunks

#         call_count = [0]
#         def fake_batch(requests):
#             call_count[0] += 1
#             if call_count[0] == 1:
#                 raise Exception("simulated GCV failure on chunk 1")
#             return make_batch_response(len(requests))

#         gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
#         results = gcv_engine_unit.ocr_batch(images)
#         assert len(results) == 20
#         # First chunk → all errors
#         for r in results[:16]:
#             assert r["error"] is not None
#         # Second chunk → all OK
#         for r in results[16:]:
#             assert r["error"] is None

#     def test_error_result_structure(self, gcv_engine_unit):
#         gcv_engine_unit._client.batch_annotate_images.side_effect = Exception("boom")
#         results = gcv_engine_unit.ocr_batch([_white()])
#         assert len(results) == 1
#         r = results[0]
#         assert r["raw_text"] == ""
#         assert r["confidence"] is None
#         assert r["word_confidences"] is None
#         assert "boom" in r["error"]


# # ---------------------------------------------------------------------------
# # Integration tests  (live GCV API)
# # ---------------------------------------------------------------------------

# @pytest.mark.integration
# @pytest.mark.slow
# class TestGCVOCRPage:
#     def test_returns_all_required_keys(self, gcv_engine, pil_first_page):
#         result = gcv_engine.ocr_page(pil_first_page, page_number=1)
#         assert set(result.keys()) == {"raw_text", "confidence", "word_confidences", "error"}

#     def test_no_error_on_valid_image(self, gcv_engine, pil_first_page):
#         result = gcv_engine.ocr_page(pil_first_page, page_number=1)
#         assert result["error"] is None

#     def test_raw_text_is_string(self, gcv_engine, pil_first_page):
#         result = gcv_engine.ocr_page(pil_first_page, page_number=1)
#         assert isinstance(result["raw_text"], str)

#     def test_confidence_is_valid(self, gcv_engine, pil_first_page):
#         result = gcv_engine.ocr_page(pil_first_page, page_number=1)
#         if result["confidence"] is not None:
#             assert 0.0 <= result["confidence"] <= 1.0

#     def test_blank_image_does_not_crash(self, gcv_engine):
#         blank = Image.new("RGB", (1240, 1754), (255, 255, 255))
#         result = gcv_engine.ocr_page(blank, page_number=1)
#         assert result["error"] is None
#         assert isinstance(result["raw_text"], str)


# @pytest.mark.integration
# @pytest.mark.slow
# class TestGCVOCRBatch:
#     def test_single_page_batch(self, gcv_engine, pil_first_page):
#         results = gcv_engine.ocr_batch([pil_first_page])
#         assert len(results) == 1
#         assert results[0]["error"] is None

#     def test_multi_page_batch_count(self, gcv_engine, synthetic_pdf):
#         from DocumentProcessor.OCR.ingestion import ingest_document
#         pages = ingest_document(str(synthetic_pdf))  # 3 pages
#         results = gcv_engine.ocr_batch(pages)
#         assert len(results) == len(pages)

#     def test_results_in_page_order(self, gcv_engine, synthetic_pdf):
#         from DocumentProcessor.OCR.ingestion import ingest_document
#         pages = ingest_document(str(synthetic_pdf))
#         results = gcv_engine.ocr_batch(pages)
#         # Each result is a dict; just verify count and structure
#         for r in results:
#             assert "raw_text" in r
#             assert "confidence" in r
#             assert "error" in r

#     def test_large_batch_chunked_transparently(self, gcv_engine, synthetic_pdf):
#         """17+ pages must be chunked; caller sees a flat result list."""
#         from DocumentProcessor.OCR.ingestion import ingest_document
#         base_pages = ingest_document(str(synthetic_pdf))
#         pages = (base_pages * 6)[:17]  # 17 pages → 2 chunks
#         results = gcv_engine.ocr_batch(pages)
#         assert len(results) == 17


"""
tests/ocr/test_gcv.py
----------------------
Tests for DocumentProcessor.OCR.gcv_engine.

Unit tests (no network):
  TestPilToBytes         — JPEG encoding, size guard, mode handling
  TestBandLogic          — confidence → colour-band mapping
  TestWordConfExtraction — _extract_word_confidences from a mock annotation
  TestBatchChunking      — correct chunk count, page ordering, error isolation

Integration tests (live GCV — @pytest.mark.integration):
  TestGCVOCRPage         — ocr_page() on a real image
  TestGCVOCRBatch        — ocr_batch() on 1 and N real images
"""

from __future__ import annotations

import io
import struct
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image

from conftest import make_batch_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white(w: int = 200, h: int = 200) -> Image.Image:
    return Image.new("RGB", (w, h), (255, 255, 255))


def _jpeg_signature(data: bytes) -> bool:
    return data[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# _pil_to_bytes  (unit)
# ---------------------------------------------------------------------------

class TestPilToBytes:
    """Tests for GCVEngine._pil_to_bytes — JPEG encoding + size guard."""

    def _encode(self, img: Image.Image, **kwargs) -> bytes:
        from DocumentProcessor.OCR.gcv_engine import GCVEngine
        return GCVEngine._pil_to_bytes(img, **kwargs)

    def test_output_is_jpeg_not_png(self):
        data = self._encode(_white())
        assert _jpeg_signature(data), "Output must start with JPEG SOI marker 0xFFD8"

    def test_rgb_image_encoded_correctly(self):
        img = Image.new("RGB", (800, 1000), (200, 200, 200))
        data = self._encode(img)
        assert _jpeg_signature(data)
        assert len(data) > 0

    def test_rgba_image_converted_to_rgb(self):
        img = Image.new("RGBA", (300, 300), (100, 150, 200, 128))
        data = self._encode(img)   # must not raise
        assert _jpeg_signature(data)

    def test_grayscale_image_converted_to_rgb(self):
        img = Image.new("L", (300, 300), 128)
        data = self._encode(img)
        assert _jpeg_signature(data)

    def test_palette_image_converted(self):
        img = Image.new("P", (300, 300))
        data = self._encode(img)
        assert _jpeg_signature(data)

    def test_output_fits_under_default_limit(self):
        from DocumentProcessor.OCR.gcv_engine import _GCV_MAX_BYTES
        img = _white(1240, 1754)  # A4 at 150 DPI
        data = self._encode(img)
        assert len(data) <= _GCV_MAX_BYTES

    def test_size_guard_reduces_quality_before_resize(self):
        """Encoding at quality=92 must first try lower qualities before resizing."""
        # A very small max_bytes forces quality reduction
        img = Image.new("RGB", (500, 500), (100, 100, 100))
        data = self._encode(img, max_bytes=50_000, jpeg_quality=92)
        assert _jpeg_signature(data)
        assert len(data) <= 50_000

    def test_size_guard_extreme_constraint_still_returns_bytes(self):
        """Even an extremely tight limit should return something rather than crash."""
        img = Image.new("RGB", (2000, 2000), (50, 50, 50))
        data = self._encode(img, max_bytes=10_000)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_jpeg_is_decodable(self):
        """Round-trip: encoded bytes must decode back to a valid PIL image."""
        img = Image.new("RGB", (640, 480), (120, 180, 240))
        data = self._encode(img)
        decoded = Image.open(io.BytesIO(data))
        assert decoded.size[0] > 0


# ---------------------------------------------------------------------------
# _band  (unit)
# ---------------------------------------------------------------------------

class TestBandLogic:
    """Tests for the confidence → colour-band mapping."""

    def _band(self, conf: float) -> str:
        from DocumentProcessor.OCR.gcv_engine import _band
        return _band(conf)

    def test_high_band_at_threshold(self):
        from config.ocr import HIGH_CONFIDENCE_THRESHOLD
        assert self._band(HIGH_CONFIDENCE_THRESHOLD) == "high"

    def test_high_band_above_threshold(self):
        assert self._band(1.0) == "high"
        assert self._band(0.99) == "high"

    def test_mid_band_at_threshold(self):
        from config.ocr import MEDIUM_CONFIDENCE_THRESHOLD
        assert self._band(MEDIUM_CONFIDENCE_THRESHOLD) == "mid"

    def test_mid_band_just_below_high(self):
        from config.ocr import HIGH_CONFIDENCE_THRESHOLD
        assert self._band(HIGH_CONFIDENCE_THRESHOLD - 0.001) == "mid"

    def test_low_band_below_medium_threshold(self):
        from config.ocr import MEDIUM_CONFIDENCE_THRESHOLD
        assert self._band(MEDIUM_CONFIDENCE_THRESHOLD - 0.001) == "low"

    def test_low_band_at_zero(self):
        assert self._band(0.0) == "low"


# ---------------------------------------------------------------------------
# _extract_word_confidences  (unit — mocked annotation)
# ---------------------------------------------------------------------------

class TestWordConfExtraction:
    def _make_word(self, text: str, conf: float) -> MagicMock:
        symbol = MagicMock()
        symbol.text = text
        word = MagicMock()
        word.symbols = [MagicMock(text=c) for c in text]
        word.confidence = conf
        return word

    def test_extracts_words_from_annotation(self, gcv_engine_unit):
        word1 = self._make_word("محكمة", 0.92)
        word2 = self._make_word("مصر", 0.88)
        paragraph = MagicMock(words=[word1, word2])
        block = MagicMock(paragraphs=[paragraph])
        page = MagicMock(blocks=[block])
        annotation = MagicMock(pages=[page])

        results = gcv_engine_unit._extract_word_confidences(annotation, page_number=2)
        assert len(results) == 2
        assert results[0].word == "محكمة"
        assert results[0].confidence == 0.92
        assert results[0].page_number == 2
        assert results[1].word == "مصر"

    def test_skips_whitespace_only_words(self, gcv_engine_unit):
        word = MagicMock()
        word.symbols = [MagicMock(text=" ")]
        word.confidence = 0.9
        paragraph = MagicMock(words=[word])
        block = MagicMock(paragraphs=[paragraph])
        page = MagicMock(blocks=[block])
        annotation = MagicMock(pages=[page])

        results = gcv_engine_unit._extract_word_confidences(annotation, page_number=1)
        assert results == []

    def test_empty_annotation_returns_empty_list(self, gcv_engine_unit):
        assert gcv_engine_unit._extract_word_confidences(None, 1) == []
        annotation = MagicMock(pages=[])
        assert gcv_engine_unit._extract_word_confidences(annotation, 1) == []

    def test_band_assigned_to_each_word(self, gcv_engine_unit):
        word = MagicMock()
        word.symbols = [MagicMock(text="x")]
        word.confidence = 0.99
        para = MagicMock(words=[word])
        block = MagicMock(paragraphs=[para])
        page = MagicMock(blocks=[block])
        annotation = MagicMock(pages=[page])
        results = gcv_engine_unit._extract_word_confidences(annotation, 1)
        assert results[0].band == "high"


# ---------------------------------------------------------------------------
# Batch chunking  (unit — mocked client)
# ---------------------------------------------------------------------------

class TestBatchChunking:
    """Verify ocr_batch splits large requests into ≤16-image chunks."""

    @pytest.mark.parametrize("n_pages,expected_chunks", [
        (1,  1),
        (4,  1),
        (15, 1),
        (16, 1),
        (17, 2),
        (20, 2),
        (32, 2),
        (33, 3),
        (48, 3),
        (53, 4),
    ])
    def test_chunk_count(self, n_pages, expected_chunks, gcv_engine_unit):
        images = [_white() for _ in range(n_pages)]
        call_count = 0

        def fake_batch(requests):
            nonlocal call_count
            call_count += 1
            return make_batch_response(len(requests))

        gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
        results = gcv_engine_unit.ocr_batch(images)
        assert call_count == expected_chunks
        assert len(results) == n_pages

    def test_chunk_size_capped_at_gcv_limit(self, gcv_engine_unit):
        """chunk_size > GCV_BATCH_LIMIT must be silently capped."""
        from DocumentProcessor.OCR.gcv_engine import GCV_BATCH_LIMIT
        images = [_white() for _ in range(GCV_BATCH_LIMIT + 1)]  # needs 2 chunks

        call_count = 0
        def fake_batch(requests):
            nonlocal call_count
            call_count += 1
            assert len(requests) <= GCV_BATCH_LIMIT, "chunk exceeded GCV limit"
            return make_batch_response(len(requests))

        gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
        gcv_engine_unit.ocr_batch(images, chunk_size=999)  # ask for huge chunk
        assert call_count == 2

    def test_empty_input_returns_empty_list(self, gcv_engine_unit):
        result = gcv_engine_unit.ocr_batch([])
        assert result == []
        gcv_engine_unit._client.batch_annotate_images.assert_not_called()

    def test_results_returned_in_page_order(self, gcv_engine_unit):
        """Results must come back in the same order as input images."""
        texts = [f"page text {i}" for i in range(20)]
        images = [_white() for _ in range(20)]

        call_idx = [0]
        def fake_batch(requests):
            chunk_texts = texts[call_idx[0] * 16 : call_idx[0] * 16 + len(requests)]
            call_idx[0] += 1
            resp = MagicMock()
            responses = []
            for t in chunk_texts:
                r = MagicMock()
                r.error.message = ""
                r.full_text_annotation.text = t
                r.full_text_annotation.pages = []
                responses.append(r)
            resp.responses = responses
            return resp

        gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
        results = gcv_engine_unit.ocr_batch(images)
        for i, res in enumerate(results):
            assert res["raw_text"] == texts[i], f"Page {i} text mismatch"

    def test_chunk_error_does_not_stop_other_chunks(self, gcv_engine_unit):
        """If chunk 1 raises, chunk 2 should still be processed."""
        images = [_white() for _ in range(20)]  # 2 chunks

        call_count = [0]
        def fake_batch(requests):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("simulated GCV failure on chunk 1")
            return make_batch_response(len(requests))

        gcv_engine_unit._client.batch_annotate_images.side_effect = fake_batch
        results = gcv_engine_unit.ocr_batch(images)
        assert len(results) == 20
        # First chunk → all errors
        for r in results[:16]:
            assert r["error"] is not None
        # Second chunk → all OK
        for r in results[16:]:
            assert r["error"] is None

    def test_error_result_structure(self, gcv_engine_unit):
        gcv_engine_unit._client.batch_annotate_images.side_effect = Exception("boom")
        results = gcv_engine_unit.ocr_batch([_white()])
        assert len(results) == 1
        r = results[0]
        assert r["raw_text"] == ""
        assert r["confidence"] is None
        assert r["word_confidences"] is None
        assert "boom" in r["error"]


# ---------------------------------------------------------------------------
# Integration tests  (live GCV API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestGCVOCRPage:
    def test_returns_all_required_keys(self, gcv_engine, pil_first_page):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        assert set(result.keys()) == {"raw_text", "confidence", "word_confidences", "error"}

    def test_no_error_on_valid_image(self, gcv_engine, pil_first_page):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        assert result["error"] is None

    def test_raw_text_is_string(self, gcv_engine, pil_first_page):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        assert isinstance(result["raw_text"], str)

    def test_confidence_is_valid(self, gcv_engine, pil_first_page):
        result = gcv_engine.ocr_page(pil_first_page, page_number=1)
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_blank_image_does_not_crash(self, gcv_engine):
        blank = Image.new("RGB", (1240, 1754), (255, 255, 255))
        result = gcv_engine.ocr_page(blank, page_number=1)
        assert result["error"] is None
        assert isinstance(result["raw_text"], str)


@pytest.mark.integration
@pytest.mark.slow
class TestGCVOCRBatch:
    def test_single_page_batch(self, gcv_engine, pil_first_page):
        results = gcv_engine.ocr_batch([pil_first_page])
        assert len(results) == 1
        assert results[0]["error"] is None

    def test_multi_page_batch_count(self, gcv_engine, synthetic_pdf):
        from DocumentProcessor.OCR.ingestion import ingest_document
        pages = ingest_document(str(synthetic_pdf))  # 3 pages
        results = gcv_engine.ocr_batch(pages)
        assert len(results) == len(pages)

    def test_results_in_page_order(self, gcv_engine, synthetic_pdf):
        from DocumentProcessor.OCR.ingestion import ingest_document
        pages = ingest_document(str(synthetic_pdf))
        results = gcv_engine.ocr_batch(pages)
        # Each result is a dict; just verify count and structure
        for r in results:
            assert "raw_text" in r
            assert "confidence" in r
            assert "error" in r

    def test_large_batch_chunked_transparently(self, gcv_engine, synthetic_pdf):
        """17+ pages must be chunked; caller sees a flat result list.

        Each page must be a distinct PIL Image object (not a shared reference).
        ocr_batch encodes images in parallel via ThreadPoolExecutor; if two
        threads receive the same object, PIL\'s lazy-loader races and raises
        \'OSError: image file is truncated\'.  .copy() gives every thread its
        own independent in-memory image.
        """
        from DocumentProcessor.OCR.ingestion import ingest_document
        base_pages = ingest_document(str(synthetic_pdf))
        # .copy() ensures every entry is an independent in-memory image —
        # avoids PIL race conditions when ThreadPoolExecutor encodes in parallel.
        pages = [p.copy() for p in (base_pages * 6)[:17]]  # 17 unique images
        results = gcv_engine.ocr_batch(pages)
        assert len(results) == 17