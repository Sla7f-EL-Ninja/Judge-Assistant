"""
Tests for the OCR pipeline production modules.

These tests cover the modules that can be tested without the QARI model
(ingestion, restoration, perspective correction, text reconstruction,
confidence scoring, and data models).
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from OCR.confidence import compute_page_confidence
from OCR.ingestion import ingest_document
from OCR.models import OCRDocumentResult, OCRPageResult, WordConfidence
from OCR.perspective_correction import (
    _check_safety_guards,
    expand_quad,
    order_points,
    perspective_correct,
)
from OCR.restoration import restore_image
from OCR.text_reconstruction import normalize_numerals, strip_html_tags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rgb_image():
    arr = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_grayscale_image():
    arr = np.random.randint(0, 255, (300, 200), dtype=np.uint8)
    return Image.fromarray(arr, "L")


@pytest.fixture
def tmp_image_file(sample_rgb_image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        sample_rgb_image.save(f, format="PNG")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_jpeg_file(sample_rgb_image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        sample_rgb_image.save(f, format="JPEG")
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_word_confidence_creation(self):
        wc = WordConfidence(word="test", confidence=0.95)
        assert wc.word == "test"
        assert wc.confidence == 0.95

    def test_ocr_page_result_defaults(self):
        page = OCRPageResult(page_number=1)
        assert page.page_number == 1
        assert page.raw_text == ""
        assert page.normalized_text == ""
        assert page.perspective_corrected is False
        assert page.confidence is None
        assert page.word_confidences is None
        assert page.error is None

    def test_ocr_page_result_full(self):
        page = OCRPageResult(
            page_number=2,
            raw_text="raw",
            normalized_text="normalized",
            perspective_corrected=True,
            confidence=0.87,
            word_confidences=[WordConfidence(word="hello", confidence=0.9)],
        )
        assert page.page_number == 2
        assert page.confidence == 0.87
        assert len(page.word_confidences) == 1

    def test_ocr_document_result(self):
        doc = OCRDocumentResult(
            metadata={"filename": "test.pdf"},
            pages=[OCRPageResult(page_number=1, raw_text="hello")],
        )
        assert doc.metadata["filename"] == "test.pdf"
        assert len(doc.pages) == 1

    def test_ocr_document_result_serialization(self):
        doc = OCRDocumentResult(
            metadata={"test": True},
            pages=[OCRPageResult(page_number=1)],
        )
        data = doc.model_dump()
        assert isinstance(data, dict)
        assert "metadata" in data
        assert "pages" in data


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_ingest_png_image(self, tmp_image_file):
        pages = ingest_document(tmp_image_file)
        assert len(pages) == 1
        assert pages[0].mode == "RGB"

    def test_ingest_jpeg_image(self, tmp_jpeg_file):
        pages = ingest_document(tmp_jpeg_file)
        assert len(pages) == 1
        assert pages[0].mode == "RGB"

    def test_ingest_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            ingest_document("/nonexistent/file.png")

    def test_ingest_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document(str(bad_file))

    def test_ingest_file_too_large(self, tmp_image_file):
        with pytest.raises(ValueError, match="exceeds limit"):
            ingest_document(tmp_image_file, max_file_size_mb=0.00001)

    def test_ingest_custom_allowed_extensions(self, tmp_image_file):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_document(tmp_image_file, allowed_extensions=[".pdf"])


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------

class TestRestoration:
    def test_restore_rgb_image(self, sample_rgb_image):
        result = restore_image(sample_rgb_image)
        assert result.mode == "RGB"
        assert result.size == sample_rgb_image.size

    def test_restore_grayscale_converts_to_rgb(self, sample_grayscale_image):
        result = restore_image(sample_grayscale_image)
        assert result.mode == "RGB"

    def test_restore_resizes_large_image(self):
        large = Image.fromarray(
            np.random.randint(0, 255, (5000, 3000, 3), dtype=np.uint8), "RGB",
        )
        result = restore_image(large, max_image_dimension=2000)
        w, h = result.size
        assert max(w, h) <= 2000

    def test_restore_preserves_small_image(self, sample_rgb_image):
        result = restore_image(sample_rgb_image, max_image_dimension=4000)
        assert result.size == sample_rgb_image.size


# ---------------------------------------------------------------------------
# Text Reconstruction
# ---------------------------------------------------------------------------

class TestTextReconstruction:
    def test_normalize_arabic_indic_numerals(self):
        text = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
        assert normalize_numerals(text) == "0123456789"

    def test_normalize_persian_numerals(self):
        text = "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9"
        assert normalize_numerals(text) == "0123456789"

    def test_normalize_mixed_text(self):
        text = "Page \u0661\u0662 of \u06F3\u06F4"
        assert normalize_numerals(text) == "Page 12 of 34"

    def test_normalize_no_change(self):
        text = "Hello world 123"
        assert normalize_numerals(text) == text

    def test_strip_html_tags(self):
        text = "<h3><i>test</i> text</h3>"
        assert strip_html_tags(text) == "test text"

    def test_strip_html_no_tags(self):
        text = "plain text"
        assert strip_html_tags(text) == text


# ---------------------------------------------------------------------------
# Perspective Correction Helpers
# ---------------------------------------------------------------------------

class TestPerspectiveCorrectionHelpers:
    def test_order_points(self):
        pts = np.array(
            [[100, 0], [0, 0], [0, 100], [100, 100]], dtype=np.float32,
        )
        ordered = order_points(pts)
        np.testing.assert_array_equal(ordered[0], [0, 0])
        np.testing.assert_array_equal(ordered[2], [100, 100])

    def test_expand_quad(self):
        pts = np.array(
            [[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32,
        )
        img_shape = (100, 100, 3)
        expanded = expand_quad(pts, img_shape, margin_pct=0.1)
        assert expanded[0][0] < pts[0][0] or expanded[0][1] < pts[0][1]

    def test_safety_guards_pass(self):
        src_pts = np.array(
            [[0, 0], [800, 0], [800, 1000], [0, 1000]], dtype=np.float32,
        )
        is_safe, reason = _check_safety_guards(
            src_pts, dst_w=800, dst_h=1000, orig_w=900, orig_h=1200,
        )
        assert is_safe
        assert reason == "ok"

    def test_safety_guards_too_small(self):
        src_pts = np.zeros((4, 2), dtype=np.float32)
        is_safe, reason = _check_safety_guards(
            src_pts, dst_w=50, dst_h=50, orig_w=1000, orig_h=1000,
        )
        assert not is_safe
        assert "too small" in reason

    def test_safety_guards_too_aggressive(self):
        src_pts = np.zeros((4, 2), dtype=np.float32)
        is_safe, reason = _check_safety_guards(
            src_pts, dst_w=200, dst_h=200, orig_w=1000, orig_h=1000,
        )
        assert not is_safe

    def test_perspective_correct_passthrough(self, sample_rgb_image):
        result, was_corrected = perspective_correct(sample_rgb_image)
        assert isinstance(result, Image.Image)
        # Small random image should pass through unchanged


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_compute_page_confidence_empty(self):
        assert compute_page_confidence((), torch.tensor([])) == 0.0

    def test_compute_page_confidence_high(self):
        vocab_size = 10
        logits = torch.zeros(1, vocab_size)
        logits[0, 5] = 10.0  # high logit for token 5
        scores = (logits,)
        generated_ids = torch.tensor([5])

        conf = compute_page_confidence(scores, generated_ids)
        assert 0.0 < conf <= 1.0
        assert conf > 0.9

    def test_compute_page_confidence_uniform(self):
        vocab_size = 10
        logits = torch.zeros(1, vocab_size)
        scores = (logits,)
        generated_ids = torch.tensor([3])

        conf = compute_page_confidence(scores, generated_ids)
        assert abs(conf - 0.1) < 0.01  # ~1/10

    def test_compute_page_confidence_multi_token(self):
        vocab_size = 10
        logits1 = torch.zeros(1, vocab_size)
        logits1[0, 2] = 10.0
        logits2 = torch.zeros(1, vocab_size)
        logits2[0, 7] = 10.0
        scores = (logits1, logits2)
        generated_ids = torch.tensor([2, 7])

        conf = compute_page_confidence(scores, generated_ids)
        assert conf > 0.9
