"""
tests/ocr/test_models.py
------------------------
Unit tests for Pydantic models: WordConfidence, OCRPageResult,
OCRDocumentResult.

All tests are pure-Python — no network calls, no file I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# WordConfidence
# ---------------------------------------------------------------------------

class TestWordConfidence:
    def _make(self, **kwargs):
        from DocumentProcessor.OCR.models import WordConfidence
        defaults = dict(word="كلمة", confidence=0.95, band="high", page_number=1)
        return WordConfidence(**{**defaults, **kwargs})

    def test_valid_creation(self):
        wc = self._make()
        assert wc.word == "كلمة"
        assert wc.confidence == 0.95
        assert wc.band == "high"
        assert wc.page_number == 1

    def test_confidence_boundary_zero(self):
        wc = self._make(confidence=0.0)
        assert wc.confidence == 0.0

    def test_confidence_boundary_one(self):
        wc = self._make(confidence=1.0)
        assert wc.confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._make(confidence=-0.001)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            self._make(confidence=1.001)

    def test_all_valid_bands(self):
        from DocumentProcessor.OCR.models import WordConfidence
        for band in ("high", "mid", "low"):
            wc = WordConfidence(word="test", confidence=0.5, band=band, page_number=1)
            assert wc.band == band

    def test_invalid_band_rejected(self):
        with pytest.raises(ValidationError):
            self._make(band="unknown")

    def test_default_page_number(self):
        from DocumentProcessor.OCR.models import WordConfidence
        wc = WordConfidence(word="x", confidence=0.8, band="mid")
        assert wc.page_number == 1

    def test_serialization_roundtrip(self):
        wc = self._make(word="محكمة", confidence=0.87, band="mid", page_number=3)
        data = wc.model_dump()
        from DocumentProcessor.OCR.models import WordConfidence
        wc2 = WordConfidence(**data)
        assert wc2 == wc


# ---------------------------------------------------------------------------
# OCRPageResult
# ---------------------------------------------------------------------------

class TestOCRPageResult:
    def _make(self, **kwargs):
        from DocumentProcessor.OCR.models import OCRPageResult
        defaults = dict(page_number=1)
        return OCRPageResult(**{**defaults, **kwargs})

    def test_canonical_text_prefers_refined(self):
        pr = self._make(normalized_text="raw", refined_text="refined")
        assert pr.canonical_text == "refined"

    def test_canonical_text_falls_back_to_normalized(self):
        pr = self._make(normalized_text="raw", refined_text="")
        assert pr.canonical_text == "raw"

    def test_canonical_text_empty_when_both_empty(self):
        pr = self._make(normalized_text="", refined_text="")
        assert pr.canonical_text == ""

    def test_default_fields(self):
        pr = self._make()
        assert pr.raw_text == ""
        assert pr.normalized_text == ""
        assert pr.refined_text == ""
        assert pr.perspective_corrected is False
        assert pr.confidence is None
        assert pr.word_confidences is None
        assert pr.error is None

    def test_error_field(self):
        pr = self._make(error="GCV returned 500")
        assert pr.error == "GCV returned 500"

    def test_confidence_stored(self):
        pr = self._make(confidence=0.9537)
        assert abs(pr.confidence - 0.9537) < 1e-6

    def test_word_confidences_stored(self):
        from DocumentProcessor.OCR.models import WordConfidence
        wc = WordConfidence(word="test", confidence=0.9, band="high", page_number=1)
        pr = self._make(word_confidences=[wc])
        assert len(pr.word_confidences) == 1
        assert pr.word_confidences[0].word == "test"

    def test_serialization_roundtrip(self):
        pr = self._make(
            page_number=2,
            raw_text="raw",
            normalized_text="norm",
            refined_text="refined",
            confidence=0.95,
        )
        data = pr.model_dump()
        from DocumentProcessor.OCR.models import OCRPageResult
        pr2 = OCRPageResult(**data)
        assert pr2.page_number == 2
        assert pr2.refined_text == "refined"


# ---------------------------------------------------------------------------
# OCRDocumentResult
# ---------------------------------------------------------------------------

class TestOCRDocumentResult:
    def _make(self, **kwargs):
        from DocumentProcessor.OCR.models import OCRDocumentResult
        defaults = dict(metadata={}, pages=[])
        return OCRDocumentResult(**{**defaults, **kwargs})

    def test_default_fields(self):
        doc = self._make()
        assert doc.metadata == {}
        assert doc.pages == []

    def test_metadata_stored(self):
        doc = self._make(metadata={"filename": "test.pdf", "total_pages": 3})
        assert doc.metadata["filename"] == "test.pdf"
        assert doc.metadata["total_pages"] == 3

    def test_pages_stored_in_order(self):
        from DocumentProcessor.OCR.models import OCRPageResult
        pages = [OCRPageResult(page_number=i) for i in range(1, 5)]
        doc = self._make(pages=pages)
        assert [p.page_number for p in doc.pages] == [1, 2, 3, 4]

    def test_model_dump_is_json_serializable(self):
        import json
        from DocumentProcessor.OCR.models import OCRPageResult
        doc = self._make(
            metadata={"filename": "x.pdf"},
            pages=[OCRPageResult(page_number=1, raw_text="hello", confidence=0.9)],
        )
        data = doc.model_dump()
        dumped = json.dumps(data, ensure_ascii=False)
        assert "x.pdf" in dumped

    def test_roundtrip_with_word_confidences(self):
        from DocumentProcessor.OCR.models import OCRDocumentResult, OCRPageResult, WordConfidence
        wc = WordConfidence(word="اختبار", confidence=0.88, band="mid", page_number=1)
        page = OCRPageResult(page_number=1, word_confidences=[wc])
        doc = OCRDocumentResult(metadata={"total_pages": 1}, pages=[page])
        data = doc.model_dump()
        doc2 = OCRDocumentResult(**data)
        assert doc2.pages[0].word_confidences[0].word == "اختبار"
